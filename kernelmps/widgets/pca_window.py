"""PCA Analysis panel — docked in main window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSplitter, QTableView, QVBoxLayout, QWidget, QMessageBox,
)

from kernelmps.models.pandas_model import PandasModel
from kernelmps.models.sort_filter_proxy import SortFilterProxy


class PCAPanel(QWidget):
    """PCA results panel — docked in the main stacked widget."""

    back_clicked = Signal()

    def __init__(self, data_path: str, result_dir: str = "", dark: bool = True, parent=None):
        super().__init__(parent)
        self._result_dir = result_dir
        self._dark = dark

        # --- Read + PCA (fast numpy) ---
        names, rows = [], []
        try:
            with open(data_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) < 2: continue
                    names.append(parts[0])
                    rows.append([float(v) for v in parts[1:]])
        except Exception as e:
            QMessageBox.critical(self, "PCA Error", str(e))
            self._ok = False
            return

        if len(rows) < 3:
            QMessageBox.warning(self, "PCA", "Need at least 3 samples.")
            self._ok = False
            return

        X = np.array(rows, dtype=np.float64)
        X_c = X - X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0  # guard against zero-variance columns
        X_s = X_c / std
        cov = np.cov(X_s, rowvar=False)
        w, v = np.linalg.eigh(cov)
        idx = np.argsort(w)[::-1]
        w, v = w[idx], v[:, idx]
        self._scores = X_s @ v
        self._var_pct = w / w.sum() * 100
        self._names = names
        self._ok = True
        self._destroyed = False
        self._mpl_cids: list = []

        self._build_ui()

    @property
    def is_ok(self) -> bool:
        return self._ok

    def _build_ui(self) -> None:
        dark = self._dark
        layout = QVBoxLayout(self)

        # Back button
        top = QHBoxLayout()
        back = QPushButton("Back to Samples")
        back.clicked.connect(self.back_clicked.emit)
        back.setFixedWidth(180)
        top.addWidget(back)
        top.addStretch()
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        # Left: scatter
        left = QGroupBox(f"PCA Scatter (PC1 {self._var_pct[0]:.1f}% vs PC2 {self._var_pct[1]:.1f}%)")
        left_splitter = QSplitter(Qt.Vertical)
        ll = QVBoxLayout(left); ll.addWidget(left_splitter)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        # --- Scatter plot ---
        scatter_w = QWidget()
        sl = QVBoxLayout(scatter_w); sl.setContentsMargins(0,0,0,0)
        self._fig = fig = Figure(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor("#2B2B2B" if dark else "#FFF")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#2B2B2B" if dark else "#FFF")

        # Distance-based coloring: light blue (near origin) → dark blue (far)
        dists = np.sqrt(self._scores[:, 0]**2 + self._scores[:, 1]**2)
        norm_dists = dists / max(dists.max(), 1e-9)
        # light=#7BAFD4 (0.48,0.69,0.83) → dark=#0A3D6B (0.04,0.24,0.42)
        r = 0.48 - 0.44 * norm_dists
        g = 0.69 - 0.45 * norm_dists
        b = 0.83 - 0.41 * norm_dists
        rgba = np.column_stack([r, g, b, np.full_like(r, 0.85)])

        scatter = ax.scatter(
            self._scores[:, 0], self._scores[:, 1],
            c=rgba, s=50, edgecolors="white", linewidth=0.5,
            picker=True, pickradius=8,
        )
        # Red origin marker
        ax.scatter([0], [0], c="#FF3333", s=100, marker="x", linewidth=2, zorder=10)

        tc = "#CCC" if dark else "#333"
        ax.set_xlabel(f"PC1 ({self._var_pct[0]:.1f}%)", color=tc, fontfamily="Times New Roman", fontsize=13)
        ax.set_ylabel(f"PC2 ({self._var_pct[1]:.1f}%)", color=tc, fontfamily="Times New Roman", fontsize=13)
        ax.tick_params(colors="#AAA" if dark else "#666")
        for s in ax.spines.values(): s.set_color("#555" if dark else "#CCC")
        ax.grid(True, alpha=0.2, color="#888")
        fig.tight_layout()
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, scatter_w)
        sl.addWidget(toolbar)
        sl.addWidget(canvas)
        left_splitter.addWidget(scatter_w)

        hover_lbl = QLabel("Hover over a point to see plant name")
        hover_lbl.setFont(QFont("Times New Roman", 13))
        hover_lbl.setStyleSheet(f"color:{tc};")
        hover_lbl.setAlignment(Qt.AlignCenter)
        ll.addWidget(hover_lbl)

        # Median chart (zoomable via QGraphicsView)
        median_lbl = QLabel()
        median_lbl.setAlignment(Qt.AlignCenter)
        from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

        class _ZoomView(QGraphicsView):
            def wheelEvent(self, ev):
                f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
                self.scale(f, f)

        med_scene = QGraphicsScene()
        med_pix = QGraphicsPixmapItem()
        med_scene.addItem(med_pix)
        med_view = _ZoomView(med_scene)
        med_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        med_view.setRenderHints(med_view.renderHints() | QPainter.Antialiasing)
        med_view.setDragMode(QGraphicsView.ScrollHandDrag)
        med_view.setMinimumHeight(120)
        med_view.setMaximumHeight(300)
        self._med_view = med_view
        self._med_pix = med_pix
        self._med_scene = med_scene
        left_splitter.addWidget(med_view)

        # Right: scores
        right = QGroupBox("PC Scores (PC1-PC5)")
        rl = QVBoxLayout(right)
        sd = {"plant_name": self._names}
        for i in range(5): sd[f"PC{i+1}"] = self._scores[:, i].round(4)
        df = pd.DataFrame(sd).sort_values("PC1", key=lambda x: x.abs(), ascending=False)
        model = PandasModel(df)
        proxy = SortFilterProxy(); proxy.setSourceModel(model)
        table = QTableView(); table.setModel(proxy)
        table.setSortingEnabled(True); table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True); table.verticalHeader().setVisible(False)
        table.resizeColumnsToContents()
        rl.addWidget(table)

        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setSizes([600, 450])
        layout.addWidget(splitter)

        # Hover
        annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#2B2B2B", ec="#555", alpha=0.9),
            color="#FFF", fontfamily="Times New Roman", fontsize=12,
            arrowprops=dict(arrowstyle="->", color="#CCC"))
        annot.set_visible(False)

        def on_hover(event):
            if self._destroyed:
                return
            if event.inaxes == ax:
                cont, d = scatter.contains(event)
                if cont and d is not None and len(d.get("ind",[])) > 0:
                    i = d["ind"][0]
                    annot.xy = (self._scores[i,0], self._scores[i,1])
                    annot.set_text(self._names[i])
                    annot.set_visible(True)
                    hover_lbl.setText(f"Plant: {self._names[i]}")
                    fig.canvas.draw_idle()
                    return
            annot.set_visible(False)
            hover_lbl.setText("Hover over a point to see plant name")
            fig.canvas.draw_idle()

        self._mpl_cids.append(fig.canvas.mpl_connect("motion_notify_event", on_hover))

        def on_click(event):
            if self._destroyed:
                return
            if event.inaxes == ax:
                cont, d = scatter.contains(event)
                if cont and d is not None and len(d.get("ind",[])) > 0:
                    i = d["ind"][0]
                    plant = self._names[i]
                    for r in range(model.rowCount()):
                        if model.data(model.index(r,0)) == plant:
                            tp = proxy.mapFromSource(model.index(r,0))
                            table.selectRow(tp.row()); table.scrollTo(tp)
                            break
                    if self._result_dir:
                        png = Path(self._result_dir) / "representative_shapes" / plant / f"{plant}_stacked_area.png"
                        if png.exists():
                            pix = QPixmap(str(png))
                            self._med_pix.setPixmap(pix)
                            self._med_scene.setSceneRect(pix.rect())
                            self._med_view.fitInView(self._med_scene.sceneRect(), Qt.KeepAspectRatio)
                            hover_lbl.setText(f"Median: {plant}")

        self._mpl_cids.append(fig.canvas.mpl_connect("button_press_event", on_click))
        canvas.draw()

    def _cleanup(self) -> None:
        self._destroyed = True
        for cid in self._mpl_cids:
            try:
                self._fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mpl_cids.clear()
