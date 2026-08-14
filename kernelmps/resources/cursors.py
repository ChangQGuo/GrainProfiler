"""App icon and corn cursor rendered from kernelmps_minifig PNGs."""

from pathlib import Path
from PySide6.QtGui import QPixmap, QCursor, QIcon, QTransform
from PySide6.QtCore import Qt

_ICON_DIR = Path(__file__).resolve().parent.parent.parent / "kernelmps_minifig"
_APP_ICON = _ICON_DIR / "图标.png"
_CURSOR_IMG = _ICON_DIR / "玉米.png"


def create_app_icon(size: int = 64) -> QIcon:
    pix = QPixmap(str(_APP_ICON))
    if pix.isNull():
        return QIcon()
    if pix.width() != size:
        pix = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QIcon(pix)


def create_corn_cursor(size: int = 48, hotspot: tuple = (4, 4)) -> QCursor:
    pix = QPixmap(str(_CURSOR_IMG))
    if pix.isNull():
        return QCursor(Qt.ArrowCursor)
    pix = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    # Flip horizontally so the corn points left (like a mouse cursor)
    flipped = pix.transformed(QTransform().scale(-1, 1))
    return QCursor(flipped, hotspot[0], hotspot[1])
