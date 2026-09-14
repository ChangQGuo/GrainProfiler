from pathlib import Path
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter, QTransform
from PySide6.QtCore import Qt

BASE = Path(__file__).resolve().parent.parent.parent
SVG = BASE / "grainprofiler_minifig"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# List SVG files
for f in SVG.iterdir():
    print(f"Found: {f.name}")

# Use glob to find files
svgs = list(SVG.glob("*.svg"))
print(f"SVGs: {svgs}")

for svg_path in svgs:
    print(f"Processing: {svg_path}")
    r = QSvgRenderer(str(svg_path))
    print(f"  Valid: {r.isValid()}")
    pix = QPixmap(256, 256)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    r.render(p)
    p.end()
    if "corn" in str(svg_path).lower() or "kernel" in str(svg_path).lower():
        pix.save(str(OUT / "grainprofiler.ico"), "ICO")
        print("  -> icon")
    else:
        # Corn plant
        flipped = pix.transformed(QTransform().scale(-1, 1))
        flipped.save(str(OUT / "corn_cursor.png"), "PNG")
        print("  -> cursor")
print("Done")
