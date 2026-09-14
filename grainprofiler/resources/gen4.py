print("1")
import os
print("2")
svg_dir = r"C:\Users\HP\Desktop\Academic Presentation\seed_project\a_aguo_test_new\grainprofiler_minifig"
print("3", os.path.exists(svg_dir))
files = os.listdir(svg_dir)
print("4", files)
