"""Trait distribution panel — box plots with outlier detection per trait."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)


class SampleAnalysisWindow(QWidget):
    """Trait distribution: box plots with outlier detection, multi-trait split view."""

    back_clicked = Signal()

    _BASE_TRAITS = [
        ("median_length_mm", "Median Length (mm)"),
        ("median_max_width_mm", "Median Max Width (mm)"),
        ("median_area_mm2", "Median Area (mm²)"),
        ("median_circularity", "Median Circularity"),
        ("median_perimeter_mm", "Median Perimeter (mm)"),
    ]

    def __init__(self, data_path: str, dark: bool = True, parent=None):
        super().__init__(parent)
        self._dark = dark
        self._ok = False

        try:
            self._df = pd.read_csv(data_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Instance copy of traits list (prevents class-variable duplication bug)
        self._traits = list(self._BASE_TRAITS)

        # Compute weight_per_kernel_g
        if "weight_g" in self._df.columns and "n_kernels" in self._df.columns:
            self._df["weight_per_kernel_g"] = (
                self._df["weight_g"] / self._df["n_kernels"].replace(0, np.nan)
            )
            self._traits.append(("weight_per_kernel_g", "Weight per Kernel (g)"))

        self._available = [(c, l) for c, l in self._traits if c in self._df.columns]
        if not self._available:
            QMessageBox.warning(self, "Analysis", "No trait columns found.")
            return

        self._ok = True
        self._build_ui()

    @property
    def is_ok(self) -> bool:
        return self._ok

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        dark = self._dark
        tc = "#CCC" if dark else "#333"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Top bar — back button only
        top = QHBoxLayout()
        back = QPushButton("Back to Samples")
        back.clicked.connect(self.back_clicked.emit)
        back.setFixedWidth(180)
        top.addWidget(back)
        top.addStretch()
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        # LEFT: trait checkboxes — dark bg + white text in dark mode
        left = QGroupBox("Traits")
        if dark:
            left.setStyleSheet("""
                QGroupBox { color: #FFF; background-color: #1A1A1A; border: 1px solid #444;
                            border-radius: 4px; margin-top: 8px; padding-top: 16px; }
                QGroupBox::title { color: #FFF; }
                QCheckBox { color: #DDD; }
                QCheckBox::indicator { border: 1px solid #888; background: #333; }
                QCheckBox::indicator:checked { background: #7BAFD4; }
                QScrollArea { background-color: #1A1A1A; border: none; }
            """)
        lscroll = QScrollArea()
        lscroll.setWidgetResizable(True)
        lw = QWidget()
        if dark:
            lw.setStyleSheet("background-color: #1A1A1A;")
        ll = QVBoxLayout(lw)
        self._checks: dict[str, QCheckBox] = {}
        for col, label in self._available:
            cb = QCheckBox(label)
            cb.stateChanged.connect(self._on_check_changed)
            self._checks[col] = cb
            ll.addWidget(cb)
        ll.addStretch()
        lscroll.setWidget(lw)
        lscroll.setMinimumWidth(220)
        lwrap = QVBoxLayout(left)
        lwrap.addWidget(lscroll)
        splitter.addWidget(left)

        # RIGHT: multi-panel chart area (horizontal for box plots)
        self._chart_splitter = QSplitter(Qt.Horizontal)
        self._chart_splitter.setMinimumWidth(500)
        splitter.addWidget(self._chart_splitter)

        splitter.setSizes([250, 700])
        layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Trait selection
    # ------------------------------------------------------------------

    def _on_check_changed(self) -> None:
        selected = [col for col, cb in self._checks.items() if cb.isChecked()]

        # Remove all existing chart panels
        while self._chart_splitter.count() > 0:
            w = self._chart_splitter.widget(0)
            w.setParent(None)
            w.deleteLater()
        self._panels: list[QWidget] = []

        if not selected:
            return

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        n = len(selected)
        label_map = dict(self._available)
        dark = self._dark
        tc = "#CCC" if dark else "#333"

        for col in selected:
            panel = QWidget()
            pl = QVBoxLayout(panel)
            pl.setContentsMargins(0, 0, 0, 0)

            fig = Figure(figsize=(3.5 if n > 2 else 5, 5.5), dpi=100)
            fig.patch.set_facecolor("#2B2B2B" if dark else "#FFF")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#2B2B2B" if dark else "#FFF")

            sub = self._df[[col, "plant_name"]].dropna().copy()
            vals = sub[col].values
            names = sub["plant_name"].values

            if len(vals) < 2:
                continue

            # Box plot stats
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outlier_mask = (vals < lo) | (vals > hi)

            # Box plot
            bp = ax.boxplot(
                vals, vert=True, patch_artist=True, widths=0.4,
                boxprops=dict(facecolor="#7BAFD4", edgecolor="#2C6FAC", linewidth=2.0),
                whiskerprops=dict(color="#2C6FAC", linewidth=2.0),
                capprops=dict(color="#2C6FAC", linewidth=2.0),
                medianprops=dict(color="black", linewidth=2.5),
                flierprops=dict(marker="", markersize=0),
            )

            # Scatter overlay
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            xs = np.full(len(vals), 1.0) + jitter

            normal = ~outlier_mask
            if np.any(normal):
                ax.scatter(
                    xs[normal], vals[normal],
                    c="#7BAFD4", s=35, alpha=0.75, edgecolors="#2C6FAC",
                    linewidth=0.5, zorder=5, picker=True, pickradius=8,
                )
            if np.any(outlier_mask):
                ax.scatter(
                    xs[outlier_mask], vals[outlier_mask],
                    c="#FF3333", s=55, alpha=0.9, edgecolors="#CC0000",
                    linewidth=1.2, zorder=6, picker=True, pickradius=10,
                )

            trait_label = label_map.get(col, col)
            ax.set_title(trait_label, color=tc, fontfamily="Times New Roman",
                         fontsize=16, fontweight="bold")
            ax.set_ylabel(trait_label, color=tc, fontfamily="Times New Roman",
                          fontsize=14, fontweight="bold")
            ax.tick_params(axis='both', labelsize=12, colors="#AAA" if dark else "#666")
            ax.set_xticks([1])
            ax.set_xticklabels([""])
            for s in ax.spines.values():
                s.set_color("#555" if dark else "#CCC")
            ax.grid(True, alpha=0.2, color="#888", axis="y")
            fig.tight_layout(pad=2.5)

            # --- hover annotation ---
            annot = ax.annotate(
                "", xy=(0, 0), xytext=(-65, 12), textcoords="offset points",
                annotation_clip=False,
                bbox=dict(boxstyle="round", fc="#2B2B2B", ec="#555", alpha=0.92),
                color="#FFF", fontfamily="Times New Roman", fontsize=11,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="#999"),
            )
            annot.set_visible(False)

            def make_on_hover(_xs=xs, _vals=vals, _names=names,
                              _annot=annot, _fig=fig, _ax=ax, _out_mask=outlier_mask):
                def on_hover(event):
                    if event.inaxes != _ax:
                        _annot.set_visible(False)
                        _fig.canvas.draw_idle()
                        return
                    hit = False
                    for coll in _ax.collections:
                        if not coll.get_picker():
                            continue
                        cont, d = coll.contains(event)
                        if cont and d is not None and len(d.get("ind", [])) > 0:
                            i = d["ind"][0]
                            offsets = coll.get_offsets()
                            dists = np.abs(_xs - offsets[i, 0]) + np.abs(
                                _vals - offsets[i, 1])
                            gi = int(np.argmin(dists))
                            if 0 <= gi < len(_names):
                                tag = "[OUTLIER] " if _out_mask[gi] else ""
                                _annot.xy = (offsets[i, 0], offsets[i, 1])
                                _annot.set_text(f"{tag}{_names[gi]}")
                                _annot.set_visible(True)
                                hit = True
                            break
                    if not hit:
                        _annot.set_visible(False)
                    _fig.canvas.draw_idle()
                return on_hover

            def make_on_click(_xs=xs, _vals=vals, _names=names,
                              _df=self._df, _ax=ax):
                def on_click(event):
                    if event.inaxes != _ax:
                        return
                    for coll in _ax.collections:
                        if not coll.get_picker():
                            continue
                        cont, d = coll.contains(event)
                        if cont and d is not None and len(d.get("ind", [])) > 0:
                            i = d["ind"][0]
                            offsets = coll.get_offsets()
                            dists = np.abs(_xs - offsets[i, 0]) + np.abs(
                                _vals - offsets[i, 1])
                            gi = int(np.argmin(dists))
                            if 0 <= gi < len(_names):
                                plant = _names[gi]
                                row = _df[_df["plant_name"] == plant]
                                if not row.empty:
                                    info = f"Plant: {plant}\n"
                                    for c in row.columns:
                                        if c != "plant_name":
                                            info += f"{c}: {row.iloc[0][c]:.4g}\n"
                                    QMessageBox.information(None, "Sample Detail", info)
                            return
                return on_click

            cid_hover = fig.canvas.mpl_connect("motion_notify_event", make_on_hover())
            cid_click = fig.canvas.mpl_connect("button_press_event", make_on_click())

            panel._mpl_cids = [cid_hover, cid_click]
            panel._fig = fig

            canvas = FigureCanvasQTAgg(fig)
            pl.addWidget(canvas)
            self._chart_splitter.addWidget(panel)
            self._panels.append(panel)

        # Auto-distribute space evenly (horizontal)
        def _resize():
            w = self._chart_splitter.width()
            if w > 10 and n > 0:
                self._chart_splitter.setSizes([w // n] * n)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, _resize)

    # ------------------------------------------------------------------
    # Cleanup on close / re-open
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        for panel in getattr(self, '_panels', []):
            if hasattr(panel, '_mpl_cids') and hasattr(panel, '_fig'):
                for cid in panel._mpl_cids:
                    try:
                        panel._fig.canvas.mpl_disconnect(cid)
                    except Exception:
                        pass

    def closeEvent(self, event) -> None:
        self._cleanup()
        super().closeEvent(event)
