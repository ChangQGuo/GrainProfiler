"""Left sidebar: Open button + searchable sample list + theme toggle."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class NavPanel(QFrame):
    """Left navigation panel."""

    folder_opened = Signal(str)
    sample_selected = Signal(str)
    theme_toggled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)
        self._sample_ids: list[str] = []
        self._dark_mode = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Open button ---
        self._open_btn = QPushButton("Open Result Folder")
        self._open_btn.clicked.connect(self._on_open_clicked)
        layout.addWidget(self._open_btn)

        # --- Path label ---
        self._path_label = QLabel("")
        self._path_label.setWordWrap(True)
        self._path_label.setObjectName("navPathLabel")
        layout.addWidget(self._path_label)

        # --- Theme toggle ---
        theme_row = QHBoxLayout()
        self._theme_btn = QPushButton("Light Theme")
        self._theme_btn.clicked.connect(self._on_theme_clicked)
        theme_row.addWidget(self._theme_btn)
        theme_row.addStretch()
        layout.addLayout(theme_row)

        # --- Filter ---
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter samples...")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self._filter_edit)

        # --- Sample list ---
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

        # --- Count label ---
        self._count_label = QLabel("")
        self._count_label.setObjectName("navCountLabel")
        layout.addWidget(self._count_label)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def populate(self, sample_ids: list[str], result_dir: str) -> None:
        self._sample_ids = sample_ids
        self._path_label.setText(result_dir)
        self._rebuild_list()

    def clear_data(self) -> None:
        self._sample_ids = []
        self._path_label.setText("")
        self._filter_edit.clear()
        self._list.clear()
        self._count_label.setText("")

    def set_theme_state(self, dark: bool) -> None:
        self._dark_mode = dark
        self._theme_btn.setText("Light Theme" if dark else "Dark Theme")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_open_clicked(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        dlg_dir = QFileDialog.getExistingDirectory(self, "Select Result Folder")
        if dlg_dir:
            self.folder_opened.emit(dlg_dir)

    def _on_theme_clicked(self) -> None:
        self.theme_toggled.emit()

    def _on_filter_changed(self, text: str) -> None:
        self._rebuild_list(text)

    def _on_item_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.sample_selected.emit(current.text())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_list(self, filter_text: str = "") -> None:
        self._list.blockSignals(True)
        self._list.clear()
        filt = filter_text.strip().lower()
        visible = [s for s in self._sample_ids if filt in s.lower()]
        for sid in visible:
            self._list.addItem(QListWidgetItem(sid))
        self._list.blockSignals(False)
        self._count_label.setText(f"{len(visible)} / {len(self._sample_ids)} samples")
        if visible:
            self._list.setCurrentRow(0)
