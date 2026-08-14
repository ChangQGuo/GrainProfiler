import os, sys
print("start")
sys.path.insert(0, r"C:\Users\HP\Desktop\Academic Presentation\seed_project\a_aguo_test_new")
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter, QTransform
from PySide6.QtCore import Qt

SVG_DIR = r"C:\Users\HP\Desktop\Academic Presentation\seed_project\a_aguo_test_new\kernelmps_minifig"
OUT_DIR = r"C:\Users\HP\Desktop\Academic Presentation\seed_project\a_aguo_test_new\kernelmps\resources"
os.makedirs(OUT_DIR, exist_ok=True)

files = os.listdir(SVG_DIR)
print("SVGs:", files)

for f in files:
    if not f.endswith('.svg'):
        continue
    path = os.path.join(SVG_DIR, f)
    print(f"Loading: {path}")
    r = QSvgRenderer(path)
    print(f"  valid={r.isValid()}")
    pix = QPixmap(256, 256)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    r.render(p)
    p.end()
    print("  rendered")
    # corn plant SVG = cursor, kernel SVG = icon
    out_path = os.path.join(OUT_DIR, "kernelmps.ico" if len(files) > 1 and f != files[0] else "corn_cursor.png")
    if out_path.endswith('.ico'):
        pix.save(out_path, 'ICO')
    else:
        flipped = pix.transformed(QTransform().scale(-1, 1))
        flipped.save(out_path, 'PNG')
    print(f"  saved: {out_path}")
print("Done!")
