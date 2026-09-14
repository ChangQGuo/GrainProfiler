print("1")
import os
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtCore import Qt
print("2 imports ok")

svg_dir = r"C:\Users\HP\Desktop\Academic Presentation\seed_project\a_aguo_test_new\grainprofiler_minifig"
files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]
print("3 files:", len(files))

for f in files:
    path = os.path.join(svg_dir, f)
    print("4 loading:", repr(path))
    r = QSvgRenderer(path)
    print("5 valid:", r.isValid())
    print("6 viewBox:", r.viewBoxF())
print("7 done")
