"""Right-side panel: sample metadata + sortable kernel measurements table + calibration."""

from __future__ import annotations

import re

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPushButton, QScrollArea, QTableView,
    QVBoxLayout, QWidget,
)

from kernelmps.app.data_loader import DataLoader
from kernelmps.models.pandas_model import PandasModel
from kernelmps.models.sort_filter_proxy import SortFilterProxy

_STYLE_DARK = """
    QWidget { background-color:#111; }
    QGroupBox { font-weight:bold; font-size:16px; color:#FFF; background-color:#111; border:1px solid #333; border-radius:0px; margin-top:0px; padding-top:16px; }
    QGroupBox::title { color:#FFF; font-size:16px; subcontrol-origin:margin; left:8px; padding:0 2px; }
    QLabel { color:#FFF; font-size:15px; background:transparent; }
    QTableView { font-size:14px; color:#FFF; background-color:#0D0D0D; gridline-color:#222; alternate-background-color:#151515; selection-background-color:#2C6FAC; }
    QHeaderView::section { background-color:#1A1A1A; color:#EEE; font-size:14px; padding:5px; border:1px solid #222; }
    QLineEdit { background-color:#1A1A1A; color:#FFF; font-size:14px; border:1px solid #333; padding:5px; }
    QPushButton { background-color:#222; color:#FFF; font-size:14px; border:1px solid #444; border-radius:3px; padding:7px; margin:4px; }
    QPushButton:hover { background-color:#333; }
"""

_STYLE_LIGHT = """
    QWidget { background-color:#FAFAFA; }
    QGroupBox { font-weight:bold; font-size:16px; color:#000; background-color:#FAFAFA; border:1px solid #DDD; border-radius:0px; margin-top:0px; padding-top:16px; }
    QGroupBox::title { color:#000; font-size:16px; subcontrol-origin:margin; left:8px; padding:0 2px; }
    QLabel { color:#000; font-size:15px; background:transparent; }
    QTableView { font-size:14px; color:#000; background-color:#FFF; gridline-color:#DDD; alternate-background-color:#F5F5F5; selection-background-color:#2C6FAC; }
    QHeaderView::section { background-color:#E8E8E8; color:#000; font-size:14px; padding:5px; border:1px solid #CCC; }
    QLineEdit { background-color:#FFF; color:#000; font-size:14px; border:1px solid #CCC; padding:5px; }
    QPushButton { background-color:#E0E0E0; color:#000; font-size:14px; border:1px solid #BBB; border-radius:3px; padding:7px; margin:4px; }
    QPushButton:hover { background-color:#D0D0D0; }
"""

_COLS_DISPLAY = [
    "kernel_name", "length_px", "max_width_px",
    "area_px2", "perimeter_px", "circularity", "length_width_ratio", "eccentricity",
]
_CAL_COL = "cal_px"


class CalPandasModel(PandasModel):
    """PandasModel that shows the calibration column in red."""

    def data(self, index, role=Qt.DisplayRole):
        col = self._df.columns[index.column()]
        if col == _CAL_COL and role == Qt.ForegroundRole:
            return QColor(255, 60, 60)
        if col == _CAL_COL and role == Qt.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if pd.isna(val) or val == 0:
                return ""
            return f"{float(val):.1f}"
        return super().data(index, role)


class SampleInfoPanel(QScrollArea):
    kernel_row_selected = Signal(str)
    kernel_row_double_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(500)
        self.setWidgetResizable(True)
        self.set_dark_theme(True)  # default dark
        self._data_loader: DataLoader | None = None
        self._current_image: str = ""
        self._kernel_names: list[str] = []
        self._table_columns_sized: bool = False

        container = QWidget()
        self.setWidget(container)
        layout = QVBoxLayout(container)

        # --- Metadata ---
        mg = QGroupBox("Sample Info")
        mg_layout = QVBoxLayout()
        mg_header = QHBoxLayout()

        # Lock toggle
        self._lock_btn = QPushButton("Locked")
        self._lock_btn.setCheckable(True)
        self._lock_btn.setChecked(True)
        self._lock_btn.setFixedWidth(100)
        self._lock_btn.toggled.connect(self._on_lock_toggled)
        mg_header.addWidget(QLabel(""))
        mg_header.addStretch()
        mg_header.addWidget(self._lock_btn)
        mg_layout.addLayout(mg_header)

        self._meta_form = QFormLayout()
        self._meta_edits: dict[str, QLineEdit | QLabel] = {}
        self._meta_readonly_vals: dict[str, str] = {}

        for key in ["plant_name", "weight_g"]:
            edit = QLineEdit("—")
            edit.setReadOnly(True)
            edit.setFont(QFont("Times New Roman", 14))
            edit.editingFinished.connect(lambda k=key: self._on_meta_edited(k))
            self._meta_edits[key] = edit
            self._meta_form.addRow(key.replace("_", " ").title() + ":", edit)

        for key in ["mm_per_px", "kernel_count"]:
            lbl = QLabel("—")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._meta_edits[key] = lbl
            self._meta_form.addRow(key.replace("_", " ").title() + ":", lbl)

        mg_layout.addLayout(self._meta_form)
        mg.setLayout(mg_layout)
        layout.addWidget(mg)

        # --- Plant Summary ---
        pg = QGroupBox("Plant Summary")
        self._plant_form = QFormLayout(pg)
        self._plant_lbls: dict[str, QLabel] = {}
        _plant_keys = [
            "n_kernels", "median_length_mm", "median_max_width_mm",
            "median_area_mm2", "median_circularity",
        ]
        for key in _plant_keys:
            lbl = QLabel("—")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._plant_lbls[key] = lbl
            self._plant_form.addRow(key.replace("_", " ").title() + ":", lbl)
        layout.addWidget(pg)

        # --- Kernel table (fills remaining space) ---
        tg = QGroupBox("Kernel Measurements")
        tvl = QVBoxLayout(tg)
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter by kernel ID...")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._on_filter)
        tvl.addWidget(self._filter_edit)

        self._table = QTableView()
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.clicked.connect(self._on_row_clicked)
        self._table.doubleClicked.connect(self._on_row_double_clicked)

        self._proxy = SortFilterProxy()
        self._pandas_model: CalPandasModel | None = None
        tvl.addWidget(self._table, stretch=1)
        layout.addWidget(tg, stretch=1)

        # --- Export ---
        eb = QPushButton("Export CSV")
        eb.clicked.connect(self.export_csv)
        layout.addWidget(eb)

    # ------------------------------------------------------------------
    def load_sample(self, image_name: str, data_loader: DataLoader) -> None:
        self._data_loader = data_loader
        self._current_image = image_name
        meta = data_loader.get_image_metadata(image_name)
        if meta is not None:
            for key, widget in self._meta_edits.items():
                val = meta.get(key, "—")
                text = str(val) if not pd.isna(val) else "—"
                if isinstance(widget, QLineEdit):
                    widget.setText(text)
                else:
                    widget.setText(text)
        self._kernel_names = data_loader.get_kernels_for_sample(image_name)
        self._meta_edits["kernel_count"].setText(str(len(self._kernel_names)))
        self._build_table(data_loader)
        self._load_plant_summary(data_loader, image_name)

    def _load_plant_summary(self, dl: DataLoader, image_name: str) -> None:
        """Populate plant-level median stats for the current image's plant."""
        plant = dl.image_to_plant.get(image_name, "")
        pm = dl.plant_medians
        if pm is None or pm.empty or not plant:
            for lbl in self._plant_lbls.values():
                lbl.setText("—")
            return
        df = pm.set_index("plant_name", drop=False)
        try:
            row = df.loc[plant]
        except KeyError:
            for lbl in self._plant_lbls.values():
                lbl.setText("—")
            return
        for key, lbl in self._plant_lbls.items():
            val = row.get(key)
            if val is None or (not isinstance(val, str) and pd.isna(val)):
                lbl.setText("—")
            elif key in ("median_length_mm", "median_max_width_mm", "median_area_mm2"):
                lbl.setText(f"{float(val):.2f}")
            elif key == "median_circularity":
                lbl.setText(f"{float(val):.4f}")
            else:
                lbl.setText(str(int(val)) if float(val) == int(float(val)) else f"{float(val):.1f}")

    def _build_table(self, dl: DataLoader) -> None:
        # Sort kernels by numeric suffix
        def _sort_key(kn):
            m = re.search(r"kernel_(\d+)", kn)
            return int(m.group(1)) if m else 0
        sorted_kernels = sorted(self._kernel_names, key=_sort_key)

        rows = []
        for kn in sorted_kernels:
            meas = dl.get_kernel_measurement(kn)
            if meas is not None:
                rows.append(meas.to_dict())
        df = pd.DataFrame(rows)
        cols = [c for c in _COLS_DISPLAY if c in df.columns]
        if not cols:
            cols = list(df.columns)
        df = df[cols].copy()

        # Insert calibration column right after length_px
        if "length_px" in df.columns:
            loc = df.columns.get_loc("length_px") + 1
            df.insert(loc, _CAL_COL, 0.0)
        else:
            df[_CAL_COL] = 0.0

        if "kernel_name" in df.columns:
            df["kernel_name"] = df["kernel_name"].apply(
                lambda x: x.split("_kernel_")[1].replace(".jpg", "")
                if "_kernel_" in str(x) else str(x)
            )

        # Save column widths before replacing model
        saved_widths = {}
        header = self._table.horizontalHeader()
        for c in range(header.count()):
            saved_widths[c] = header.sectionSize(c)

        self._pandas_model = CalPandasModel(df)
        self._proxy.setSourceModel(self._pandas_model)
        self._table.setModel(self._proxy)

        if not self._table_columns_sized:
            self._table.resizeColumnsToContents()
            self._table_columns_sized = True
        elif saved_widths:
            # Restore user-adjusted widths
            for c, w in saved_widths.items():
                if c < header.count():
                    header.resizeSection(c, w)

    def set_dark_theme(self, dark: bool) -> None:
        self.setStyleSheet(_STYLE_DARK if dark else _STYLE_LIGHT)

    # ------------------------------------------------------------------
    # Editable sample info
    # ------------------------------------------------------------------

    def _on_lock_toggled(self, checked: bool) -> None:
        self._lock_btn.setText("Locked" if checked else "Unlocked")
        for key in ("plant_name", "weight_g"):
            w = self._meta_edits.get(key)
            if isinstance(w, QLineEdit):
                w.setReadOnly(checked)

    def _on_meta_edited(self, key: str) -> None:
        """Sync edit back to metadata.csv and related files."""
        if self._data_loader is None or not self._current_image:
            return
        w = self._meta_edits.get(key)
        if not isinstance(w, QLineEdit):
            return
        new_val = w.text().strip()
        if not new_val or new_val == "—":
            return
        dl = self._data_loader
        image = self._current_image

        # Update metadata.csv in memory
        if not dl.metadata.empty and "image_name" in dl.metadata.columns:
            mask = dl.metadata["image_name"] == image
            if mask.any() and key in dl.metadata.columns:
                try:
                    if key == "weight_g":
                        dl.metadata.loc[mask, key] = float(new_val)
                    else:
                        dl.metadata.loc[mask, key] = new_val
                except ValueError:
                    return
            # Write back to disk
            import os
            csv_path = os.path.join(dl.result_dir, "metadata.csv")
            dl.metadata.to_csv(csv_path, index=False)

        # If plant_name changed, update image_to_plant mapping
        if key == "plant_name":
            old_plant = dl.image_to_plant.get(image, "")
            dl.image_to_plant[image] = new_val
            # Update latent_traits.csv if present
            if not dl.latent_traits.empty:
                lt_path = os.path.join(dl.result_dir, "latent_traits.csv")
                if "plant_id" in dl.latent_traits.columns:
                    dl.latent_traits.loc[dl.latent_traits["plant_id"] == old_plant, "plant_id"] = new_val
                    dl.latent_traits.to_csv(lt_path, index=False, float_format='%.6f')

    def update_calibration(self, cal_data: dict[str, float]) -> None:
        if self._pandas_model is None:
            return
        df = self._pandas_model._df
        for r in range(len(df)):
            display = str(df.iat[r, 0])
            # Full kernel name
            stem = self._current_image.replace(".jpg", "")
            full = f"{stem}_kernel_{display}.jpg"
            if full in cal_data:
                df.iat[r, df.columns.get_loc(_CAL_COL)] = cal_data[full]
        self._pandas_model.dataChanged.emit(
            self._pandas_model.index(0, 0),
            self._pandas_model.index(self._pandas_model.rowCount() - 1,
                                     self._pandas_model.columnCount() - 1),
        )

    def select_kernel(self, kernel_name: str) -> None:
        if self._pandas_model is None:
            return
        suffix = kernel_name.split("_kernel_")[1].replace(".jpg", "") if "_kernel_" in kernel_name else kernel_name
        for r in range(self._pandas_model.rowCount()):
            idx = self._pandas_model.index(r, 0)
            val = str(self._pandas_model.data(idx, Qt.DisplayRole))
            if suffix in val:
                proxy_idx = self._proxy.mapFromSource(idx)
                self._table.selectRow(proxy_idx.row())
                self._table.scrollTo(proxy_idx)
                break

    def export_csv(self) -> None:
        if self._pandas_model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"{self._current_image}_measurements.csv",
            "CSV Files (*.csv)"
        )
        if path:
            self._pandas_model._df.to_csv(path, index=False)

    def _on_filter(self, text: str) -> None:
        self._proxy.set_filter(text)

    def _on_row_clicked(self, index) -> None:
        if self._data_loader is None or self._pandas_model is None:
            return
        src = self._proxy.mapToSource(index)
        row_data = self._pandas_model.get_row_data(src.row())
        if row_data is not None:
            dn = str(row_data.get("kernel_name", ""))
            stem = self._current_image.replace(".jpg", "")
            full = f"{stem}_kernel_{dn}.jpg"
            self.kernel_row_selected.emit(full)

    def _on_row_double_clicked(self, index) -> None:
        if self._data_loader is None or self._pandas_model is None:
            return
        src = self._proxy.mapToSource(index)
        row_data = self._pandas_model.get_row_data(src.row())
        if row_data is not None:
            dn = str(row_data.get("kernel_name", ""))
            stem = self._current_image.replace(".jpg", "")
            full = f"{stem}_kernel_{dn}.jpg"
            self.kernel_row_double_clicked.emit(full)
