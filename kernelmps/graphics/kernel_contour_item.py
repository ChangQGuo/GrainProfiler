"""Interactive kernel contour polygon with hover / click support."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QToolTip,
)

from kernelmps.app.settings import (
    CONTOUR_HOVER,
    CONTOUR_NORMAL,
    CONTOUR_SELECTED,
    CONTOUR_WIDTH_HOVER,
    CONTOUR_WIDTH_NORMAL,
    CONTOUR_WIDTH_SELECTED,
)

OnClicked = Callable[[str], None]


class KernelContourItem(QGraphicsPathItem):
    """A single kernel contour polygon that responds to hover and click."""

    def __init__(
        self,
        kernel_name: str,
        path: QPainterPath,
        tooltip_data: dict[str, str],
        on_clicked: OnClicked | None = None,
        on_hovered: OnClicked | None = None,
        on_unhovered: Callable[[], None] | None = None,
        parent=None,
    ):
        super().__init__(path, parent)
        self.kernel_name = kernel_name
        self._tooltip_data = tooltip_data
        self._on_clicked = on_clicked
        self._on_hovered = on_hovered
        self._on_unhovered = on_unhovered
        self._is_selected = False
        self._remote_highlighted = False

        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QColor(*CONTOUR_NORMAL), CONTOUR_WIDTH_NORMAL))
        self.setBrush(QBrush(QColor(0, 180, 0, 40)))  # semi-transparent green fill
        self.setZValue(1)

    # ------------------------------------------------------------------
    # Hover
    # ------------------------------------------------------------------

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if not self._is_selected:
            self.setPen(QPen(QColor(*CONTOUR_HOVER), CONTOUR_WIDTH_HOVER))
        tip = (
            f"Kernel: {self.kernel_name}\n"
            f"Length: {self._tooltip_data.get('length', '-')}\n"
            f"Max Width: {self._tooltip_data.get('max_width', '-')}\n"
            f"Area: {self._tooltip_data.get('area', '-')}\n"
            f"Circularity: {self._tooltip_data.get('circularity', '-')}"
        )
        QToolTip.showText(event.screenPos().toPoint(), tip)
        if self._on_hovered:
            self._on_hovered(self.kernel_name)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if not self._is_selected and not self._remote_highlighted:
            self.setPen(QPen(QColor(*CONTOUR_NORMAL), CONTOUR_WIDTH_NORMAL))
        QToolTip.hideText()
        if self._on_unhovered:
            self._on_unhovered()

    # ------------------------------------------------------------------
    # Click
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._on_clicked is not None:
            self._on_clicked(self.kernel_name)
            event.accept()
        else:
            super().mousePressEvent(event)

    # ------------------------------------------------------------------
    # Selection state
    # ------------------------------------------------------------------

    def set_selected(self, selected: bool) -> None:
        self._is_selected = selected
        if selected:
            pen = QPen(QColor(*CONTOUR_SELECTED), CONTOUR_WIDTH_SELECTED)
            self.setPen(pen)
            self.setZValue(2)
        else:
            self.setPen(QPen(QColor(*CONTOUR_NORMAL), CONTOUR_WIDTH_NORMAL))
            self.setZValue(1)

    def set_remote_highlighted(self, v: bool) -> None:
        self._remote_highlighted = v
        if v and not self._is_selected:
            self.setPen(QPen(QColor(*CONTOUR_HOVER), CONTOUR_WIDTH_HOVER))
        elif not v and not self._is_selected:
            self.setPen(QPen(QColor(*CONTOUR_NORMAL), CONTOUR_WIDTH_NORMAL))

    def set_outlined(self, v: bool) -> None:
        """Mark this contour as an outlier with a red border."""
        if v:
            self.setPen(QPen(QColor(255, 51, 51), 3.0))
            self.setBrush(QBrush(QColor(255, 0, 0, 50)))
            self.setZValue(5)
        else:
            if not self._is_selected:
                self.setPen(QPen(QColor(*CONTOUR_NORMAL), CONTOUR_WIDTH_NORMAL))
                self.setBrush(QBrush(QColor(0, 180, 0, 40)))
                self.setZValue(1)

    def is_selected(self) -> bool:
        return self._is_selected
