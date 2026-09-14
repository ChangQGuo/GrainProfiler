"""VAE Latent Explorer — interactive latent-space shape exploration."""

from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)


def _get_onnx_dir() -> str:
    """Locate ONNX model files (works both dev and frozen)."""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'onnx_models')
    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project, 'onnx_models')


class VAELatentWindow(QWidget):
    """VAE latent explorer panel — docked in the main stacked widget."""

    back_clicked = Signal()

    _LATENT_RANGE = 3.5        # ±3.5 sigma
    _LATENT_STEP = 0.01
    _SLIDER_MULT = 100         # slider uses int, spinbox uses float

    def __init__(self, data_path: str, latent_csv: str, dark: bool = True, parent=None):
        super().__init__(parent)
        self._dark = dark
        self._ok = False

        # --- Load ONNX decoder ---
        try:
            import onnxruntime as ort
            onnx_dir = _get_onnx_dir()
            onnx_path = os.path.join(onnx_dir, 'profile_vae_latent5.onnx')
            self._ort_session = ort.InferenceSession(onnx_path)
            self._col_mean = np.load(os.path.join(onnx_dir, 'vae_col_mean.npy')).squeeze()
            self._col_std = np.load(os.path.join(onnx_dir, 'vae_col_std.npy')).squeeze()
        except Exception as e:
            QMessageBox.critical(self, "VAE Error", f"Failed to load ONNX model:\n{e}")
            return

        # --- Load data ---
        self._names: list[str] = []
        self._profiles: dict[str, np.ndarray] = {}    # raw 100-dim full-width
        self._latents: dict[str, list[float]] = {}     # original 5-dim latent values

        try:
            self._load_data(data_path, latent_csv)
        except Exception as e:
            QMessageBox.critical(self, "VAE Error", str(e))
            return

        if len(self._names) < 1:
            QMessageBox.warning(self, "VAE", "No valid profiles found.")
            return

        self._current_name: str = ""
        self._orig_latents: list[float] = []
        self._updating_ui: bool = False  # guard against recursive signal loops

        self._ok = True
        self._build_ui()

    @property
    def is_ok(self) -> bool:
        return self._ok

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, profile_txt: str, latent_csv: str) -> None:
        """Read width profiles and latent traits."""
        # Profiles
        with open(profile_txt, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                vals = np.array([float(v) for v in parts[1:]], dtype=np.float64)
                if len(vals) >= 2:
                    self._names.append(name)
                    self._profiles[name] = vals

        self._names.sort()

        # Latent traits
        import pandas as pd
        try:
            df = pd.read_csv(latent_csv)
            for _, row in df.iterrows():
                plant = str(row.iloc[0])
                vals = [float(row.iloc[i + 1]) for i in range(5)]
                self._latents[plant] = vals
        except Exception:
            pass  # latent traits optional — just use zeros

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        dark = self._dark
        tc = "#CCC" if dark else "#333"
        layout = QVBoxLayout(self)

        # --- Top bar ---
        top = QHBoxLayout()
        back = QPushButton("Back to Samples")
        back.clicked.connect(self.back_clicked.emit)
        back.setFixedWidth(180)
        top.addWidget(back)
        top.addStretch()
        layout.addLayout(top)

        # --- Three-panel splitter ---
        splitter = QSplitter(Qt.Horizontal)

        # LEFT: plant list
        left = QGroupBox("Plants")
        ll = QVBoxLayout(left)
        self._list = QListWidget()
        self._list.addItems(self._names)
        self._list.currentTextChanged.connect(self._on_plant_selected)
        ll.addWidget(self._list)
        splitter.addWidget(left)

        # CENTER: profile comparison chart
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, 4), dpi=100)
        fig.patch.set_facecolor("#2B2B2B" if dark else "#FFF")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#2B2B2B" if dark else "#FFF")
        ax.set_xlabel("Position along normalized axis (%)", color=tc, fontfamily="Times New Roman", fontsize=14)
        ax.set_ylabel("Full Width (scaled)", color=tc, fontfamily="Times New Roman", fontsize=14)
        ax.tick_params(colors="#AAA" if dark else "#666")
        for s in ax.spines.values():
            s.set_color("#555" if dark else "#CCC")
        ax.grid(True, alpha=0.2, color="#888")
        fig.tight_layout(pad=2)
        self._fig = fig
        self._ax = ax
        self._canvas = FigureCanvasQTAgg(fig)
        splitter.addWidget(self._canvas)

        # RIGHT: latent controls
        right = QGroupBox("Latent Dimensions (±3.5σ)")
        rl = QVBoxLayout(right)
        note = QLabel("Latents ~ N(0,1); ±3.5σ covers 99.95% of population")
        note.setFont(QFont("Times New Roman", 10))
        note.setStyleSheet(f"color:{tc}; font-style: italic;")
        note.setWordWrap(True)
        rl.addWidget(note)

        self._sliders: list[QSlider] = []
        self._spinboxes: list[QDoubleSpinBox] = []
        self._dim_lbls: list[QLabel] = []

        for i in range(5):
            dim_box = QGroupBox(f"Latent {i + 1}")
            dim_layout = QVBoxLayout(dim_box)

            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-int(self._LATENT_RANGE * self._SLIDER_MULT),
                            int(self._LATENT_RANGE * self._SLIDER_MULT))
            slider.setSingleStep(1)
            slider.setPageStep(10)
            slider.valueChanged.connect(lambda v, idx=i: self._on_slider_changed(idx, v))
            dim_layout.addWidget(slider)

            # Spinbox + label row
            spin_row = QHBoxLayout()
            val_lbl = QLabel("0.00")
            val_lbl.setFont(QFont("Times New Roman", 12))
            val_lbl.setStyleSheet(f"color:#7BAFD4;" if dark else "color:#2C6FAC;")
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setMinimumWidth(60)
            spin_row.addWidget(val_lbl)

            spin = QDoubleSpinBox()
            spin.setRange(-self._LATENT_RANGE, self._LATENT_RANGE)
            spin.setSingleStep(self._LATENT_STEP)
            spin.setDecimals(2)
            spin.valueChanged.connect(lambda v, idx=i: self._on_spinbox_changed(idx, v))
            spin_row.addWidget(spin)
            dim_layout.addLayout(spin_row)

            self._sliders.append(slider)
            self._spinboxes.append(spin)
            self._dim_lbls.append(val_lbl)
            rl.addWidget(dim_box)

        # Reset button
        reset_btn = QPushButton("Reset to Original")
        reset_btn.clicked.connect(self._on_reset)
        rl.addWidget(reset_btn)
        rl.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([200, 550, 250])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Plant selection
    # ------------------------------------------------------------------

    def _on_plant_selected(self, name: str) -> None:
        if not name:
            return
        self._current_name = name
        raw_profile = self._profiles.get(name)

        # Get latent values (use zeros if not found)
        latents = self._latents.get(name, [0.0] * 5)
        if len(latents) < 5:
            latents = [0.0] * 5
        self._orig_latents = list(latents)

        # Update sliders without triggering decode
        self._updating_ui = True
        for i in range(5):
            val = latents[i]
            self._sliders[i].blockSignals(True)
            self._spinboxes[i].blockSignals(True)
            self._sliders[i].setValue(int(val * self._SLIDER_MULT))
            self._spinboxes[i].setValue(val)
            self._dim_lbls[i].setText(f"{val:.2f}")
            self._sliders[i].blockSignals(False)
            self._spinboxes[i].blockSignals(False)
        self._updating_ui = False

        # Decode and plot
        recon = self._decode(latents)
        self._plot_profiles(raw_profile, recon)

    # ------------------------------------------------------------------
    # ONNX decode
    # ------------------------------------------------------------------

    def _decode(self, latents: list[float]) -> np.ndarray:
        """Run ONNX decoder: 5-dim z → denormalized 100-dim profile."""
        z = np.array(latents, dtype=np.float32).reshape(1, 5)
        recon_norm = self._ort_session.run(None, {'latent': z})[0]
        recon_raw = recon_norm.squeeze() * self._col_std + self._col_mean
        return recon_raw

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_profiles(self, original: np.ndarray | None, reconstructed: np.ndarray) -> None:
        ax = self._ax
        ax.clear()
        dark = self._dark
        tc = "#CCC" if dark else "#333"
        xs = np.arange(100)

        if original is not None:
            ax.plot(xs, original, color="#2C6FAC", linewidth=2.0, label="Original")

        ax.plot(xs, reconstructed, color="#E0853A", linewidth=2.0, linestyle="--", label="VAE Recon")

        ax.set_xlabel("Position along normalized axis (%)", color=tc, fontfamily="Times New Roman", fontsize=14)
        ax.set_ylabel("Full Width (scaled)", color=tc, fontfamily="Times New Roman", fontsize=14)
        ax.tick_params(colors="#AAA" if dark else "#666")
        for s in ax.spines.values():
            s.set_color("#555" if dark else "#CCC")
        ax.grid(True, alpha=0.2, color="#888")
        ax.legend(loc="upper right", facecolor="#333" if dark else "#EEE",
                  edgecolor="#555", labelcolor=tc, fontsize=10)
        self._fig.tight_layout(pad=2)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Slider / Spinbox sync
    # ------------------------------------------------------------------

    def _on_slider_changed(self, idx: int, slider_val: int) -> None:
        if self._updating_ui:
            return
        val = slider_val / self._SLIDER_MULT
        self._updating_ui = True
        self._spinboxes[idx].blockSignals(True)
        self._spinboxes[idx].setValue(val)
        self._spinboxes[idx].blockSignals(False)
        self._dim_lbls[idx].setText(f"{val:.2f}")
        self._updating_ui = False
        self._decode_and_redraw()

    def _on_spinbox_changed(self, idx: int, val: float) -> None:
        if self._updating_ui:
            return
        self._updating_ui = True
        self._sliders[idx].blockSignals(True)
        self._sliders[idx].setValue(int(val * self._SLIDER_MULT))
        self._sliders[idx].blockSignals(False)
        self._dim_lbls[idx].setText(f"{val:.2f}")
        self._updating_ui = False
        self._decode_and_redraw()

    def _decode_and_redraw(self) -> None:
        latents = [sb.value() for sb in self._spinboxes]
        recon = self._decode(latents)
        raw = self._profiles.get(self._current_name)
        self._plot_profiles(raw, recon)

    def _on_reset(self) -> None:
        if not self._orig_latents:
            return
        self._updating_ui = True
        for i in range(5):
            val = self._orig_latents[i]
            self._sliders[i].blockSignals(True)
            self._spinboxes[i].blockSignals(True)
            self._sliders[i].setValue(int(val * self._SLIDER_MULT))
            self._spinboxes[i].setValue(val)
            self._dim_lbls[i].setText(f"{val:.2f}")
            self._sliders[i].blockSignals(False)
            self._spinboxes[i].blockSignals(False)
        self._updating_ui = False
        self._decode_and_redraw()
