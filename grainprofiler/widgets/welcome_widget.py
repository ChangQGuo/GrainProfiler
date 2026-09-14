"""Welcome / placeholder widget shown before a result folder is opened."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class WelcomeWidget(QFrame):
    """Centered call-to-action shown at startup."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.NoFrame)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("GrainProfiler")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Times New Roman", 38, QFont.Bold))

        subtitle = QLabel("Maize Kernel Phenotyping Result Inspector")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Times New Roman", 18))

        hint = QLabel("Open a result folder to begin  ->  File  |  Open Result Folder")
        hint.setAlignment(Qt.AlignCenter)
        hint.setObjectName("welcomeHint")
        hint.setFont(QFont("Times New Roman", 14))

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(16)
        layout.addWidget(hint)
