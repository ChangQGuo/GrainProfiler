print("starting...")
try:
    from PySide6.QtSvg import QSvgRenderer
    print("QSvgRenderer imported")
    from PySide6.QtGui import QPixmap, QPainter
    print("QPixmap imported")
    from PySide6.QtCore import Qt
    print("Qt imported")
except Exception as e:
    print(f"Error: {e}")
