"""QGraphicsView: tray photo + contours + nav arrows + calibration ruler."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath,
    QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from grainprofiler.app.data_loader import DataLoader
from grainprofiler.graphics.kernel_contour_item import KernelContourItem
from grainprofiler.utils.coordinate_transform import transform_contour
from grainprofiler.utils.image_conversion import cv2_to_qpixmap, load_image
from grainprofiler.utils.photo_finder import find_photo


class CalibrationLine(QGraphicsLineItem):
    """A single calibration measurement line. Click to select/delete."""

    COLOR = QColor(255, 255, 0)
    WIDTH = 3

    def __init__(self, x1, y1, x2, y2, px_len, parent=None):
        super().__init__(x1, y1, x2, y2, parent)
        self.px_len = px_len
        self.setPen(QPen(self.COLOR, self.WIDTH))
        self.setZValue(20)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

    def hoverEnterEvent(self, event):
        self.setPen(QPen(QColor(255, 80, 80), self.WIDTH + 1))

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(self.COLOR, self.WIDTH))


class SampleView(QGraphicsView):
    kernel_selected = Signal(str)
    kernel_hovered = Signal(str)
    navigate_prev = Signal()
    navigate_next = Signal()
    calibration_changed = Signal(dict)  # {kernel_name: px_length, ...}

    ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setBackgroundBrush(QBrush(QColor(45, 45, 45)))
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self._photo_pixmap: QPixmap | None = None
        self._contour_items: dict[str, KernelContourItem] = {}
        self._overlay_items: list = []
        self._selected_kernel: str | None = None
        self._highlighted_kernel: str | None = None
        self._show_overlays: bool = True
        self._data_loader: DataLoader | None = None
        self._current_image: str = ""
        self._img_h: int = 0
        self._img_w: int = 0

        # Calibration state
        self._cal_mode: bool = False
        self._cal_draw_mode: bool = True   # True=draw, False=pan
        self._cal_drawing: bool = False
        self._cal_start: QPointF | None = None
        self._cal_temp_line: QGraphicsLineItem | None = None
        self._cal_lines: list[CalibrationLine] = []
        self._cal_labels: list[QGraphicsSimpleTextItem] = []
        self._cal_kernel_map: dict[str, float] = {}  # kernel_name → px

        # Navigation arrows
        self._nav_left: QGraphicsPolygonItem | None = None
        self._nav_right: QGraphicsPolygonItem | None = None
        self._nav_left_bg: QGraphicsPolygonItem | None = None
        self._nav_right_bg: QGraphicsPolygonItem | None = None

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.ZOOM_FACTOR, self.ZOOM_FACTOR)
        else:
            self.scale(1.0 / self.ZOOM_FACTOR, 1.0 / self.ZOOM_FACTOR)

    # ------------------------------------------------------------------
    # Mouse (calibration-aware)
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if self._cal_mode and self._cal_draw_mode and event.button() == Qt.LeftButton:
            self._cal_drawing = True
            self._cal_start = self.mapToScene(event.pos())
            self._cal_temp_line = self._scene.addLine(
                0, 0, 0, 0, QPen(QColor(255, 255, 0, 100), 2, Qt.DashLine)
            )
            event.accept()
            return
        # Check viewport-relative nav arrows
        if event.button() == Qt.LeftButton and not self._cal_mode:
            p = event.pos()
            vw = self.viewport().width()
            m = 8; aw = 55; ah = 80
            cy = self.viewport().height() // 2
            if p.x() < m + aw and abs(p.y() - cy) < ah // 2:
                self.navigate_prev.emit(); event.accept(); return
            if p.x() > vw - m - aw and abs(p.y() - cy) < ah // 2:
                self.navigate_next.emit(); event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._cal_mode and self._cal_drawing and self._cal_temp_line:
            end = self.mapToScene(event.pos())
            self._cal_temp_line.setLine(
                self._cal_start.x(), self._cal_start.y(),
                end.x(), end.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._cal_mode and self._cal_drawing and event.button() == Qt.LeftButton:
            self._cal_drawing = False
            if self._cal_temp_line:
                self._scene.removeItem(self._cal_temp_line)
                self._cal_temp_line = None
            end = self.mapToScene(event.pos())
            dx = end.x() - self._cal_start.x()
            dy = end.y() - self._cal_start.y()
            px_len = np.sqrt(dx * dx + dy * dy)
            if px_len < 3:
                event.accept()
                return

            cl = CalibrationLine(
                self._cal_start.x(), self._cal_start.y(),
                end.x(), end.y(), px_len
            )
            self._scene.addItem(cl)
            self._cal_lines.append(cl)

            mid = QPointF((self._cal_start.x() + end.x()) / 2,
                          (self._cal_start.y() + end.y()) / 2)
            lbl = QGraphicsSimpleTextItem(f"{px_len:.1f} px")
            lbl.setFont(QFont("Times New Roman", 14, QFont.Bold))
            lbl.setBrush(QBrush(QColor(255, 255, 0)))
            lbl.setPos(mid)
            lbl.setZValue(25)
            self._scene.addItem(lbl)
            self._cal_labels.append(lbl)

            self._associate_calibration(cl, px_len)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _associate_calibration(self, cl: CalibrationLine, px_len: float) -> None:
        """Find closest kernel contour to the calibration line midpoint."""
        mx = (cl.line().x1() + cl.line().x2()) / 2
        my = (cl.line().y1() + cl.line().y2()) / 2
        best_kn = None
        best_dist = float("inf")
        for kn, citem in self._contour_items.items():
            cp = citem.boundingRect().center()
            dist = np.sqrt((cp.x() - mx) ** 2 + (cp.y() - my) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_kn = kn
        if best_kn is not None and best_dist < 200:
            self._cal_kernel_map[best_kn] = px_len
        else:
            # Use index as key for unattached lines
            self._cal_kernel_map[f"_line_{len(self._cal_lines)}"] = px_len
        self.calibration_changed.emit(dict(self._cal_kernel_map))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_sample(self, image_name: str, data_loader: DataLoader) -> None:
        self._data_loader = data_loader
        self._current_image = image_name
        self._clear()

        photo_path = find_photo(image_name, data_loader.result_dir)
        if photo_path is None:
            self._show_error(f"Photo not found for {image_name}")
            return
        cv_img = load_image(photo_path)
        if cv_img is None:
            self._show_error(f"Failed to load: {photo_path}")
            return

        pixmap = cv2_to_qpixmap(cv_img)
        self._photo_pixmap = pixmap
        self._img_h, self._img_w = cv_img.shape[:2]

        scene = self._scene
        scene.setSceneRect(0, 0, self._img_w, self._img_h)
        scene.addItem(QGraphicsPixmapItem(pixmap))

        # Label
        meta = data_loader.get_image_metadata(image_name)
        pn = str(meta.get("plant_name", "?")) if meta is not None else "?"
        wt = str(meta.get("weight_g", "?")) if meta is not None else "?"
        label = QGraphicsSimpleTextItem(f"  {pn}  |  {wt}g  ")
        label.setZValue(10)
        label.setFont(QFont("Times New Roman", 22, QFont.Bold))
        label.setBrush(QBrush(QColor(255, 255, 255)))
        lr = QRectF(label.boundingRect()); lr.moveTo(8, 8); label.setPos(8, 8)
        bg = QGraphicsRectItem(lr.adjusted(-6, -4, 6, 6))
        bg.setBrush(QBrush(QColor(0, 0, 0, 160)))
        bg.setPen(Qt.NoPen); bg.setZValue(9)
        scene.addItem(bg); self._overlay_items.append(bg)
        scene.addItem(label); self._overlay_items.append(label)

        # Contours
        for kn in data_loader.get_kernels_for_sample(image_name):
            cd = data_loader.get_kernel_contour(kn)
            box = data_loader.get_kernel_box(kn)
            if cd is None or box is None:
                continue
            pts = transform_contour(cd.contour, box, (self._img_h, self._img_w))
            if len(pts) < 3:
                continue
            path = QPainterPath()
            path.moveTo(pts[0][0], pts[0][1])
            for px, py in pts[1:]:
                path.lineTo(px, py)
            path.closeSubpath()
            meas = data_loader.get_kernel_measurement(kn)
            td = {}
            if meas is not None:
                td = {
                    "length": f"{meas.get('length_px', 0):.1f} px",
                    "max_width": f"{meas.get('max_width_px', 0):.1f} px",
                    "area": f"{meas.get('area_px2', 0):.0f} px2",
                    "circularity": f"{meas.get('circularity', 0):.4f}",
                }
            citem = KernelContourItem(kn, path, td,
                                      on_clicked=self._on_kernel_clicked,
                                      on_hovered=self._on_contour_hovered,
                                      on_unhovered=self._on_contour_unhovered)
            scene.addItem(citem)
            self._contour_items[kn] = citem
            self._overlay_items.append(citem)

        self._set_overlay_visibility(self._show_overlays)
        self._build_nav_arrows()
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def set_overlays_visible(self, v: bool) -> None:
        self._show_overlays = v
        self._set_overlay_visibility(v)

    def select_kernel(self, kn: str | None) -> None:
        if self._selected_kernel and self._selected_kernel in self._contour_items:
            self._contour_items[self._selected_kernel].set_selected(False)
        self._selected_kernel = kn
        if kn and kn in self._contour_items:
            self._contour_items[kn].set_selected(True)
            self.centerOn(self._contour_items[kn])

    def highlight_kernel(self, kn: str | None) -> None:
        if self._highlighted_kernel and self._highlighted_kernel in self._contour_items:
            self._contour_items[self._highlighted_kernel].set_remote_highlighted(False)
        self._highlighted_kernel = kn
        if kn and kn in self._contour_items:
            self._contour_items[kn].set_remote_highlighted(True)

    def remove_kernel(self, kn: str) -> None:
        """Remove a single kernel contour from the view."""
        if kn in self._contour_items:
            item = self._contour_items.pop(kn)
            if item in self._overlay_items:
                self._overlay_items.remove(item)
            self._scene.removeItem(item)
            if self._selected_kernel == kn:
                self._selected_kernel = None
            if self._highlighted_kernel == kn:
                self._highlighted_kernel = None

    def set_outlier_kernels(self, outliers: set[str]) -> None:
        """Mark kernel contours as outliers (red border)."""
        for kn, citem in self._contour_items.items():
            if kn in outliers:
                citem.set_outlined(True)
            else:
                citem.set_outlined(False)

    def clear_highlight(self) -> None:
        self.highlight_kernel(None)

    def set_calibration_mode(self, enabled: bool) -> None:
        self._cal_mode = enabled
        if enabled:
            self._cal_draw_mode = True
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)
            self._clear_calibration()

    def toggle_cal_draw_mode(self) -> bool:
        """Toggle between draw/pan. Returns new draw_mode state."""
        self._cal_draw_mode = not self._cal_draw_mode
        if self._cal_draw_mode:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)
        return self._cal_draw_mode

    def delete_selected_line(self, scene_pos: QPointF) -> bool:
        """Delete a calibration line near scene_pos. Returns True if deleted."""
        for i, cl in enumerate(self._cal_lines):
            l = cl.line()
            # Check distance from point to line segment
            d = self._point_line_dist(scene_pos, l.p1(), l.p2())
            if d < 15:
                self._scene.removeItem(cl)
                if i < len(self._cal_labels):
                    self._scene.removeItem(self._cal_labels[i])
                    del self._cal_labels[i]
                del self._cal_lines[i]
                self._rebuild_cal_map()
                self.calibration_changed.emit(dict(self._cal_kernel_map))
                return True
        return False

    def _point_line_dist(self, p: QPointF, a: QPointF, b: QPointF) -> float:
        ab = b - a; ap = p - a
        t = max(0.0, min(1.0, (ap.x() * ab.x() + ap.y() * ab.y()) / max(1e-9, ab.x()**2 + ab.y()**2)))
        closest = a + t * ab
        return np.sqrt((p.x() - closest.x())**2 + (p.y() - closest.y())**2)

    def _rebuild_cal_map(self) -> None:
        self._cal_kernel_map.clear()
        for cl in self._cal_lines:
            self._associate_calibration(cl, cl.px_len)

    def undo_calibration_line(self) -> None:
        if self._cal_lines:
            cl = self._cal_lines.pop()
            self._scene.removeItem(cl)
        if self._cal_labels:
            ll = self._cal_labels.pop()
            self._scene.removeItem(ll)
        self._rebuild_cal_map()
        self.calibration_changed.emit(dict(self._cal_kernel_map))

    def get_calibration_data(self) -> dict[str, float]:
        return dict(self._cal_kernel_map)

    # ------------------------------------------------------------------
    # Navigation arrows
    # ------------------------------------------------------------------
    def _build_nav_arrows(self) -> None:
        self.viewport().installEventFilter(self)

    def drawForeground(self, painter, rect):
        """Paint viewport-relative nav arrows (always same size regardless of zoom)."""
        vp = self.viewport()
        vw, vh = vp.width(), vp.height()
        m = 8; aw = 55; ah = 80; cy = vh // 2
        painter.save()
        painter.resetTransform()  # switch to viewport pixel coords
        painter.setPen(QPen(QColor(255, 255, 255, 120), 2))
        painter.setBrush(QBrush(QColor(255, 255, 255, 60)))
        lp = QPolygonF([QPointF(m+aw, cy-ah/2), QPointF(m+aw, cy+ah/2), QPointF(m, cy)])
        painter.drawPolygon(lp)
        rp = QPolygonF([QPointF(vw-m-aw, cy-ah/2), QPointF(vw-m-aw, cy+ah/2), QPointF(vw-m, cy)])
        painter.drawPolygon(rp)
        painter.restore()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj == self.viewport() and not self._cal_mode:
            if event.type() in (QEvent.Enter, QEvent.Leave, QEvent.MouseMove):
                self.viewport().update()  # trigger redraw for arrow hover effect
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _clear(self) -> None:
        self._contour_items.clear()
        self._overlay_items.clear()
        self._cal_lines.clear()
        self._cal_labels.clear()
        self._cal_kernel_map.clear()
        self._scene.clear()
        self._photo_pixmap = None

    def _clear_calibration(self) -> None:
        for cl in self._cal_lines:
            self._scene.removeItem(cl)
        for ll in self._cal_labels:
            self._scene.removeItem(ll)
        self._cal_lines.clear()
        self._cal_labels.clear()
        self._cal_kernel_map.clear()
        self._cal_drawing = False
        if self._cal_temp_line:
            self._scene.removeItem(self._cal_temp_line)
            self._cal_temp_line = None

    def _set_overlay_visibility(self, v: bool) -> None:
        for item in self._overlay_items:
            item.setVisible(v)

    def _on_kernel_clicked(self, kn: str) -> None:
        self.select_kernel(kn)
        self.kernel_selected.emit(kn)

    def _on_contour_hovered(self, kn: str) -> None:
        self.kernel_hovered.emit(kn)

    def _on_contour_unhovered(self) -> None:
        self.kernel_hovered.emit("")

    def _show_error(self, msg: str) -> None:
        t = self._scene.addSimpleText(msg)
        t.setBrush(QColor(255, 80, 80))
        t.setZValue(100)
        self._scene.setSceneRect(0, 0, 400, 200)
