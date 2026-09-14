# -*- mode: python ; coding: utf-8 -*-
# grainprofiler.spec — PyInstaller build spec
#
# Build:
#   pyinstaller grainprofiler.spec
#
# Output will be in dist/grainprofiler.exe

import sys
from pathlib import Path

_project = Path(r"C:\Users\HP\Desktop\Academic Presentation\seed_project\a_aguo_test_new")
_grainprofiler = _project / "grainprofiler"

a = Analysis(
    [str(_grainprofiler / "main.py")],
    pathex=[str(_project)],
    binaries=[],
    datas=[
        (str(_project / "grainprofiler_minifig"), "grainprofiler_minifig"),
        (str(_project / "onnx_models"), "onnx_models"),
    ],
    hiddenimports=[
        # PySide6
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtSvg",
        # Data
        "pandas",
        "numpy",
        "numpy.core._methods",
        # OpenCV (headless — no Qt conflict)
        "cv2",
        # matplotlib
        "matplotlib",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.figure",
        # ONNX runtime
        "onnxruntime",
        # stdlib that PyInstaller sometimes misses
        "json",
        "csv",
        "pathlib",
        "dataclasses",
        "re",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "ultralytics",
        "paddle", "paddleocr", "shapely", "scipy",
        "sklearn",
        "tensorflow", "keras",
        # Exclude unused matplotlib backends
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backends.backend_wxagg",
        "matplotlib.backends.backend_gtk3agg",
        "matplotlib.backends.backend_gtk4agg",
        "matplotlib.backends.backend_macosx",
        "matplotlib.backends.backend_webagg",
        "matplotlib.backends.backend_nbagg",
        "matplotlib.backends.backend_template",
        "tkinter", "_tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="grainprofiler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,           # GUI app — no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
