import os
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter, QTransform
from PySide6.QtCore import Qt

svg_dir = r"C:\Users\HP\Desktop\Academic Presentation\seed_project\a_aguo_test_new\grainprofiler_minifig"
out_dir = r"C:\Users\HP\Desktop\Academic Presentation\seed_project\a_aguo_test_new\grainprofiler\resources"
os.makedirs(out_dir, exist_ok=True)

files = sorted([f for f in os.listdir(svg_dir) if f.endswith('.svg')])
# files[0] should be sweet corn kernel (shorter name), files[1] is corn plant
for i, f in enumerate(files):
    path = os.path.join(svg_dir, f)
    r = QSvgRenderer(path)
    print(f"Processing: {f[:20]}...")

    pix = QPixmap(256, 256)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    r.render(p)
    p.end()

    if i == 0:
        # Sweet corn kernel -> app icon
        out = os.path.join(out_dir, "grainprofiler.ico")
        pix.save(out, 'ICO')
    else:
        # Corn plant -> cursor (flipped)
        flipped = pix.transformed(QTransform().scale(-1, 1))
        out = os.path.join(out_dir, "corn_cursor.png")
        flipped.save(out, 'PNG')
    print(f"  -> {out} ({os.path.getsize(out)} bytes)")

print("Done!")
