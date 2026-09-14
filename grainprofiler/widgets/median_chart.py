"""Interactive kernel width profile chart (median view)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class MedianChart(QWidget):
    """Matplotlib chart: gray lines = individual kernels, blue = median."""

    kernel_hovered = Signal(str)
    kernel_clicked = Signal(str)

    def __init__(self, profiles: dict, image_name: str, dark: bool = True, parent=None):
        super().__init__(parent)
        self._profiles = profiles
        self._image_name = image_name
        self._dark = dark
        self._highlighted_line: object | None = None
        self._lines: list = []
        self._all_names: list[str] = []
        self._destroyed: bool = False
        self._mpl_cids: list = []
        self._build()

    def _build(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        dark = self._dark
        all_names = list(self._profiles.keys())
        all_vals = list(self._profiles.values())
        xs = range(len(all_vals[0]))

        self._all_names = all_names
        self._ax = None
        self._fig = fig = Figure(figsize=(8, 3), dpi=100)
        fig.patch.set_facecolor("#2B2B2B" if dark else "#FFF")
        ax = fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor("#2B2B2B" if dark else "#FFF")

        lines = []
        for vals in all_vals:
            half = [v / 2.0 for v in vals]
            l, = ax.plot(xs, half, color="#555", linewidth=0.5, alpha=0.6, picker=5)
            lines.append(l)
        self._lines = lines

        # Median half-width (bold blue)
        arr = np.array(all_vals) / 2.0
        median = np.median(arr, axis=0)
        ax.plot(xs, median, color="#2C6FAC", linewidth=2.5)

        tc = "#CCC" if dark else "#333"
        ax.set_xlabel("Position along axis (%)", color=tc, fontfamily="Times New Roman")
        ax.set_ylabel("Half-Width (px)", color=tc, fontfamily="Times New Roman")
        ax.tick_params(colors="#AAA" if dark else "#666")
        for s in ax.spines.values():
            s.set_color("#555" if dark else "#CCC")
        ax.set_title(f"Half-Width Profiles — {self._image_name}", color=tc,
                      fontfamily="Times New Roman", fontsize=13)
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(220)
        self.setMinimumHeight(240)
        self._canvas = canvas

        self._info = info = QLabel("Hover over a line to identify kernel")
        info.setFont(QFont("Times New Roman", 12))
        info.setStyleSheet(f"color:{tc};")
        info.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        layout.addWidget(info)

        # --- hover logic ---
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(8, -16), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#333", ec="#555", alpha=0.9),
            color="#FFF", fontfamily="Times New Roman", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#AAA"),
        )
        annot.set_visible(False)
        self._annot = annot

        def on_hover(event):
            if self._destroyed:
                return
            if event.inaxes != ax:
                annot.set_visible(False)
                info.setText("Hover over a line to identify kernel")
                for i, l_ in enumerate(lines):
                    if self._highlighted_line is not lines[i]:
                        l_.set_color("#555"); l_.set_linewidth(0.5); l_.set_zorder(1)
                self.kernel_hovered.emit("")
                fig.canvas.draw_idle()
                return
            hit = False
            for i, l_ in enumerate(lines):
                if l_.contains(event)[0]:
                    hit = True
                    annot.xy = (event.xdata, event.ydata)
                    kn = all_names[i]
                    short = kn.split("_kernel_")[1].replace(".jpg", "") if "_kernel_" in kn else kn
                    annot.set_text(f"Kernel {short}")
                    annot.set_visible(True)
                    info.setText(f"Kernel: {kn}")
                    l_.set_color("#E05555"); l_.set_linewidth(2); l_.set_zorder(10)
                    self.kernel_hovered.emit(kn)
                else:
                    if self._highlighted_line is not lines[i]:
                        l_.set_color("#555"); l_.set_linewidth(0.5); l_.set_zorder(1)
            if not hit:
                annot.set_visible(False)
                info.setText("Hover over a line to identify kernel")
            fig.canvas.draw_idle()

        def on_click(event):
            if self._destroyed:
                return
            if event.inaxes != ax:
                return
            for i, l_ in enumerate(lines):
                if l_.contains(event)[0]:
                    self.kernel_clicked.emit(all_names[i])
                    return

        self._mpl_cids.append(fig.canvas.mpl_connect("motion_notify_event", on_hover))
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

    # ------------------------------------------------------------------
    # Programmatic highlight / select
    # ------------------------------------------------------------------

    def highlight_kernel(self, kn: str | None) -> None:
        if self._destroyed:
            return
        if self._highlighted_line is not None:
            self._highlighted_line.set_color("#555")
            self._highlighted_line.set_linewidth(0.5)
            self._highlighted_line.set_zorder(1)
            self._highlighted_line = None
        if kn and kn in self._all_names:
            i = self._all_names.index(kn)
            if i < len(self._lines):
                self._highlighted_line = self._lines[i]
                self._highlighted_line.set_color("#E05555")
                self._highlighted_line.set_linewidth(2)
                self._highlighted_line.set_zorder(10)
                self._info.setText(f"Kernel: {kn}")
        else:
            self._info.setText("Hover over a line to identify kernel")
        if hasattr(self, '_canvas'):
            self._canvas.draw_idle()

    def clear_highlight(self) -> None:
        self.highlight_kernel(None)
