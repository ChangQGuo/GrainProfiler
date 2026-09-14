"""Bottom (red) and top (magenta) axis endpoint markers."""

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QGraphicsEllipseItem

from grainprofiler.app.settings import BOTTOM_COLOR, TOP_COLOR


def make_endpoint(
    center: QPointF,
    radius: float,
    is_bottom: bool,
    parent=None,
) -> QGraphicsEllipseItem:
    """Create a filled circle marker for a kernel axis endpoint.

    Args:
        center: centre of the circle in scene coords.
        radius: circle radius in px.
        is_bottom: True → red bottom marker, False → magenta top marker.
    """
    r = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
    item = QGraphicsEllipseItem(r, parent)
    color = BOTTOM_COLOR if is_bottom else TOP_COLOR
    item.setBrush(QBrush(QColor(*color)))
    item.setPen(QColor(*color))  # no outline — use fill as pen so it's visible
    item.setAcceptHoverEvents(False)
    item.setZValue(2)  # above contours for visibility
    return item
