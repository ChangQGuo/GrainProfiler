"""Application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from grainprofiler.app.main_window import MainWindow

_ICON_PNG = (
    Path(__file__).resolve().parent.parent
    / "grainprofiler_minifig" / "图标.png"
)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("grainprofiler")
    app.setOrganizationName("GrainProfiler")

    # Splash screen
    splash_pix = QPixmap(str(_ICON_PNG))
    if not splash_pix.isNull():
        splash_pix = splash_pix.scaled(
            400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        splash = QSplashScreen(splash_pix)
        splash.show()
        app.processEvents()
        QTimer.singleShot(1500, splash.close)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
