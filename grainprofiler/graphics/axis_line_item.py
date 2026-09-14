"""Axis line overlay from kernel bottom to top tip."""

from PySide6.QtCore import QLineF, QPointF
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsLineItem

from grainprofiler.app.settings import AXIS_COLOR, AXIS_WIDTH


class AxisLineItem(QGraphicsLineItem):
    """Purple axis-line segment — transparent to mouse events."""

    def __init__(self, bottom: QPointF, top: QPointF, parent=None):
        super().__init__(QLineF(bottom, top), parent)
        self.setPen(QPen(QColor(*AXIS_COLOR), AXIS_WIDTH))
        self.setAcceptHoverEvents(False)
        self.setZValue(0.5)  # below contours
