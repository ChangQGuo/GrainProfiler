"""Generate app icon and cursor from SVG sources."""

from pathlib import Path
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter, QTransform
from PySide6.QtCore import Qt
import sys

BASE = Path(__file__).resolve().parent.parent.parent
SVG_DIR = BASE / "grainprofiler_minifig"
OUT_DIR = Path(__file__).resolve().parent

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- App Icon (sweet corn kernel) ---
r1 = QSvgRenderer(str(SVG_DIR / "甜玉米粒.svg"))
assert r1.isValid(), "Failed to load sweet corn SVG"
pix = QPixmap(256, 256)
pix.fill(Qt.transparent)
p = QPainter(pix)
r1.render(p)
p.end()
pix.save(str(OUT_DIR / "grainprofiler.ico"), "ICO")
print(f"Icon: {OUT_DIR / 'grainprofiler.ico'}")

# --- Corn cursor (flipped left) ---
r2 = QSvgRenderer(str(SVG_DIR / "玉米.svg"))
assert r2.isValid(), "Failed to load corn SVG"
pix2 = QPixmap(48, 48)
pix2.fill(Qt.transparent)
p2 = QPainter(pix2)
r2.render(p2)
p2.end()
flipped = pix2.transformed(QTransform().scale(-1, 1))
flipped.save(str(OUT_DIR / "corn_cursor.png"), "PNG")
print(f"Cursor: {OUT_DIR / 'corn_cursor.png'}")
print("Done!")
