"""Modal dialog showing full detail for a single kernel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from kernelmps.app.data_loader import DataLoader
from kernelmps.app.settings import (
    AXIS_COLOR,
    BOTTOM_COLOR,
    CONTOUR_NORMAL,
    TOP_COLOR,
)
from kernelmps.utils.image_conversion import cv2_to_qpixmap, load_image


class KernelDetailDialog(QDialog):
    """Per-kernel detail view with subimage, measurements, and axis overlay."""

    def __init__(
        self,
        kernel_name: str,
        data_loader: DataLoader,
        result_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Kernel Detail — {kernel_name}")
        self.setMinimumSize(950, 600)
        self._kernel_name = kernel_name
        self._data_loader = data_loader
        self._result_dir = result_dir

        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        main_layout = QHBoxLayout(self)

        # --- Left: Subimage with overlays ---
        left_group = QGroupBox("Kernel Image")
        left_layout = QVBoxLayout(left_group)
        self._view = QGraphicsView()
        self._view.setRenderHints(
            self._view.renderHints()
            | QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
        )
        self._view.setMinimumWidth(400)
        self._view.setBackgroundBrush(QBrush(QColor(45, 45, 45)))
        self._scene = QGraphicsScene()
        self._view.setScene(self._scene)
        left_layout.addWidget(self._view)
        main_layout.addWidget(left_group, stretch=2)

        # --- Right: Measurements table ---
        right_group = QGroupBox("Measurements")
        right_layout = QVBoxLayout(right_group)
        self._meas_layout = right_layout
        main_layout.addWidget(right_group, stretch=1)

        self._load_data()

    def _load_data(self) -> None:
        kn = self._kernel_name
        dl = self._data_loader

        # --- Subimage ---
        sub_path = Path(self._result_dir) / "subimages" / kn
        img_np = load_image(str(sub_path))
        if img_np is None:
            label = QLabel(f"Subimage not found:\n{sub_path}")
            label.setAlignment(Qt.AlignCenter)
            self._scene.addWidget(label)
            self._scene.setSceneRect(0, 0, 300, 200)
        else:
            h, w = img_np.shape[:2]
            pixmap = cv2_to_qpixmap(img_np)
            self._scene.addItem(QGraphicsPixmapItem(pixmap))
            self._scene.setSceneRect(0, 0, w, h)

            # --- Contour overlay ---
            contour_data = dl.get_kernel_contour(kn)
            if contour_data is not None:
                pts = contour_data.contour
                if len(pts) >= 3:
                    path = QPainterPath()
                    path.moveTo(pts[0][0], pts[0][1])
                    for px, py in pts[1:]:
                        path.lineTo(px, py)
                    path.closeSubpath()
                    ci = self._scene.addPath(
                        path,
                        QPen(QColor(*CONTOUR_NORMAL), 1.5),
                        QBrush(QColor(0, 180, 0, 50)),
                    )
                    ci.setZValue(1)

            # --- Axis overlay ---
            axis_data = dl.get_kernel_axis(kn)
            if axis_data is not None:
                bx, by = axis_data.bottom
                tx, ty = axis_data.top
                self._scene.addLine(
                    bx, by, tx, ty,
                    QPen(QColor(*AXIS_COLOR), 1),
                )
                r = 3
                for px, py, color in [
                    (bx, by, BOTTOM_COLOR),
                    (tx, ty, TOP_COLOR),
                ]:
                    dp = QPainterPath()
                    dp.addEllipse(QPointF(px, py), r, r)
                    self._scene.addPath(
                        dp, Qt.NoPen, QBrush(QColor(*color))
                    )

            self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

        # --- Measurements table ---
        meas = dl.get_kernel_measurement(kn)
        axis_data = dl.get_kernel_axis(kn)

        rows: list[dict] = []
        if meas is not None:
            for col in meas.index:
                if col != "kernel_name":
                    val = meas[col]
                    rows.append({"Metric": col, "Value": f"{val:.4f}" if isinstance(val, float) else str(val)})
        if axis_data is not None:
            rows.append({"Metric": "shape_label", "Value": axis_data.shape_label or "Normal"})

        if not rows:
            self._meas_layout.addWidget(QLabel("No measurement data available"))
            return

        df = pd.DataFrame(rows)
        # Build a simple key-value display
        # Use QFormLayout within the right group
        form = QFormLayout()
        for _, row in df.iterrows():
            form.addRow(
                QLabel(str(row["Metric"]).replace("_", " ").title()),
                QLabel(str(row["Value"])),
            )
        self._meas_layout.addLayout(form)
        self._meas_layout.addStretch()
