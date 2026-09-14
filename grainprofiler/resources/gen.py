from pathlib import Path
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter, QTransform
from PySide6.QtCore import Qt

BASE = Path(__file__).resolve().parent.parent.parent
SVG = BASE / "grainprofiler_minifig"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Icon
r1 = QSvgRenderer(str(SVG / "甜玉米粒.svg"))
pix = QPixmap(256, 256); pix.fill(Qt.transparent)
p = QPainter(pix); r1.render(p); p.end()
pix.save(str(OUT / "grainprofiler.ico"), "ICO")
print("Icon OK")

# Cursor (flipped left)
r2 = QSvgRenderer(str(SVG / "玉米.svg"))
pix2 = QPixmap(48, 48); pix2.fill(Qt.transparent)
p2 = QPainter(pix2); r2.render(p2); p2.end()
flipped = pix2.transformed(QTransform().scale(-1, 1))
flipped.save(str(OUT / "corn_cursor.png"), "PNG")
print("Cursor OK")
