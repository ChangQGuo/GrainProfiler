"""RMSE similarity box plot — outlier detection for kernel width profiles."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QMenu,
    QVBoxLayout,
    QWidget,
)

BOX_FILL = "#6BAED6"
BOX_EDGE = "#2171B5"
SCATTER_NORMAL = "#6BAED6"
SCATTER_NORMAL_EDGE = "#2171B5"
SCATTER_OUTLIER = "#E03131"
SCATTER_OUTLIER_EDGE = "#B71C1C"
MEDIAN_LINE = "#0D0D0D"


class SimilarityBoxPlot(QWidget):
    """Box plot of per-kernel RMSE vs median profile, with outlier detection."""

    kernel_hovered = Signal(str)
    kernel_clicked = Signal(str)
    kernel_delete_requested = Signal(str)

    def __init__(self, profiles: dict, image_name: str, dark: bool = True, parent=None):
        super().__init__(parent)
        self._profiles = profiles
        self._image_name = image_name
        self._dark = dark
        self._destroyed = False
        self._mpl_cids: list = []
        self._outlier_mask = None
        self._rmse_values = None
        self._kernel_names = None
        self._last_annot_text: str = ""
        self._last_annot_visible: bool = False
        self._build()

    def _build(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        dark = self._dark
        all_names = list(self._profiles.keys())
        all_vals = [self._profiles[n] for n in all_names]

        arr = np.array(all_vals) / 2.0
        median = np.median(arr, axis=0)
        rmse_vals = np.array([np.sqrt(np.mean((a - median) ** 2)) for a in arr])
        self._rmse_values = rmse_vals
        self._kernel_names = all_names

        q1 = float(np.percentile(rmse_vals, 25))
        q3 = float(np.percentile(rmse_vals, 75))
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        self._outlier_mask = (rmse_vals < lo) | (rmse_vals > hi)

        fig = Figure(figsize=(5, 5.5), dpi=100)
        bg = "#1E1E1E" if dark else "#FFF"
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        self._ax = ax
        self._fig = fig
        ax.set_facecolor(bg)

        tc = "#E0E0E0" if dark else "#222"

        ax.boxplot(
            rmse_vals, vert=True, patch_artist=True, widths=0.38,
            boxprops=dict(facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=2.2),
            whiskerprops=dict(color=BOX_EDGE, linewidth=2.0),
            capprops=dict(color=BOX_EDGE, linewidth=2.0),
            medianprops=dict(color=MEDIAN_LINE, linewidth=2.8),
            flierprops=dict(marker="", markersize=0),
        )

        xs_jitter = np.random.default_rng(42).uniform(-0.09, 0.09, len(rmse_vals))
        xs = 1.0 + xs_jitter

        normal_mask = ~self._outlier_mask
        if np.any(normal_mask):
            ax.scatter(
                xs[normal_mask], rmse_vals[normal_mask],
                c=SCATTER_NORMAL, s=42, alpha=0.78, edgecolors=SCATTER_NORMAL_EDGE,
                linewidth=0.6, zorder=5, picker=True, pickradius=8,
                label="Normal",
            )

        if np.any(self._outlier_mask):
            ax.scatter(
                xs[self._outlier_mask], rmse_vals[self._outlier_mask],
                c=SCATTER_OUTLIER, s=65, alpha=0.92, edgecolors=SCATTER_OUTLIER_EDGE,
                linewidth=1.5, zorder=6, picker=True, pickradius=10,
                label="Outlier",
            )

        ax.set_ylabel("RMSE (px)", fontsize=18, fontweight="bold", color=tc,
                       fontfamily="Times New Roman")
        ax.set_title(self._image_name,
                     fontsize=16, fontweight="bold", color=tc,
                     fontfamily="Times New Roman")
        ax.tick_params(axis='both', labelsize=13, colors="#AAA" if dark else "#555")
        ax.set_xticks([1])
        ax.set_xticklabels([""], fontsize=13)
        for s in ax.spines.values():
            s.set_color("#444" if dark else "#BBB")
        ax.grid(True, alpha=0.18, color="#666" if dark else "#CCC", axis="y")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=11,
                  prop={"weight": "bold"},
                  facecolor="#2A2A2A" if dark else "#FFF",
                  edgecolor="#555" if dark else "#CCC",
                  labelcolor=tc)

        fig.tight_layout(pad=2.5)

        annot = ax.annotate(
            "", xy=(0, 0), xytext=(-55, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=2", fc="#2A2A2A", ec="#555", alpha=0.92),
            color="#FFF", fontfamily="Times New Roman", fontsize=9,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color="#999"),
            annotation_clip=False,
        )
        annot.set_visible(False)
        self._annot = annot

        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(280)
        self._canvas = canvas

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)

        # ------ scroll-to-zoom ------
        def on_scroll(event):
            if self._destroyed:
                return
            try:
                if event.inaxes != ax:
                    return
                ylo, yhi = ax.get_ylim()
                cy = event.ydata if event.ydata is not None else (ylo + yhi) / 2
                scale = 0.85 if event.step > 0 else 1.0 / 0.85
                new_lo = cy - (cy - ylo) * scale
                new_hi = cy + (yhi - cy) * scale
                ax.set_ylim(new_lo, new_hi)
                fig.canvas.draw_idle()
            except Exception:
                pass

        self._mpl_cids.append(fig.canvas.mpl_connect("scroll_event", on_scroll))

        # ------ hover (with draw throttling) ------
        def on_hover(event):
            if self._destroyed:
                return
            try:
                if event.inaxes != ax:
                    if self._last_annot_visible:
                        annot.set_visible(False)
                        self._last_annot_visible = False
                        self._last_annot_text = ""
                        self.kernel_hovered.emit("")
                        fig.canvas.draw_idle()
                    return

                hit = False
                for coll in ax.collections:
                    if not coll.get_picker():
                        continue
                    cont, d = coll.contains(event)
                    if cont and d is not None and len(d.get("ind", [])) > 0:
                        i = d["ind"][0]
                        offsets = coll.get_offsets()
                        dists = (np.abs(xs - offsets[i, 0]) +
                                 np.abs(rmse_vals - offsets[i, 1]))
                        gi = int(np.argmin(dists))
                        if 0 <= gi < len(all_names):
                            kn = all_names[gi]
                            short = (kn.split("_kernel_")[1].replace(".jpg", "")
                                     if "_kernel_" in kn else kn)
                            tag = "[OUTLIER] " if self._outlier_mask[gi] else ""
                            new_text = f"{tag}Kernel {short}\nRMSE = {rmse_vals[gi]:.2f} px"
                            annot.xy = (offsets[i, 0], offsets[i, 1])

                            if (new_text != self._last_annot_text or
                                    not self._last_annot_visible):
                                annot.set_text(new_text)
                                annot.set_visible(True)
                                self._last_annot_text = new_text
                                self._last_annot_visible = True
                                self.kernel_hovered.emit(kn)
                                fig.canvas.draw_idle()
                            hit = True
                        break

                if not hit and self._last_annot_visible:
                    annot.set_visible(False)
                    self._last_annot_visible = False
                    self._last_annot_text = ""
                    self.kernel_hovered.emit("")
                    fig.canvas.draw_idle()
            except Exception:
                pass

        self._mpl_cids.append(fig.canvas.mpl_connect("motion_notify_event", on_hover))

        # ------ click (left = select, right = delete menu) ------
        def on_click(event):
            if self._destroyed:
                return
            try:
                if event.inaxes != ax:
                    return
                from matplotlib.backend_bases import MouseButton
                is_right = (event.button == MouseButton.RIGHT)

                for coll in ax.collections:
                    if not coll.get_picker():
                        continue
                    cont, d = coll.contains(event)
                    if cont and d is not None and len(d.get("ind", [])) > 0:
                        i = d["ind"][0]
                        offsets = coll.get_offsets()
                        dists = (np.abs(xs - offsets[i, 0]) +
                                 np.abs(rmse_vals - offsets[i, 1]))
                        gi = int(np.argmin(dists))
                        if 0 <= gi < len(all_names):
                            kn = all_names[gi]
                            if is_right:
                                short = (kn.split("_kernel_")[1].replace(".jpg", "")
                                         if "_kernel_" in kn else kn)
                                menu = QMenu(self)
                                delete_action = menu.addAction(
                                    f"Delete Kernel {short}")
                                # Use safe position fallback
                                try:
                                    pos = event.guiEvent.globalPos()
                                except Exception:
                                    from PySide6.QtGui import QCursor
                                    pos = QCursor.pos()
                                action = menu.exec(pos)
                                if action == delete_action:
                                    self.kernel_delete_requested.emit(kn)
                            else:
                                self.kernel_clicked.emit(kn)
                        return
            except Exception:
                pass

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
        self._fig = None
        self._ax = None
        self._annot = None

    @property
    def outlier_kernels(self) -> list[str]:
        if self._outlier_mask is None or self._kernel_names is None:
            return []
        return [self._kernel_names[i] for i, v in enumerate(self._outlier_mask) if v]
