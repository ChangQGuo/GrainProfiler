# Profile VAE — Unsupervised Shape Trait Discovery via Width Profiles

## Motivation

The pipeline's Stage 5 outputs `rep_width_profiles.txt`: one 100-dimensional vector per
plant, representing the median full-width profile after axis-length normalisation.
These vectors encode the *shape* of each genotype independent of absolute kernel size.

A β-Variational Autoencoder (β-VAE) compresses each 100-dimensional profile into a
**5-dimensional latent vector** `z ∈ ℝ⁵`.  The five latent dimensions serve as
**data-driven, unsupervised shape traits** suitable for:

- GWAS / QTL mapping (phenotype = latent_i)
- Genotype clustering (PCA / t-SNE / UMAP on latent space)
- Heritability estimation
- Multi-trait genomic prediction

This approach is inspired by representation-learning phenotyping strategies used in
human genomics (e.g., unsupervised traits from medical imaging for respiratory and
circulatory disease GWAS).

---

## Architecture

```
                    ┌────────── Encoder ──────────┐
  x ∈ ℝ¹⁰⁰  ──→  Linear(100,64) → BN → LReLU → Drop(0.1)
                → Linear(64, 32)  → BN → LReLU → Drop(0.1)
                → Linear(32, 16)  → BN → LReLU
                → μ  = Linear(16, 5)
                → log σ² = Linear(16, 5)
                    └──────────────────────────────┘
                              │
                              ▼  z = μ + σ·ε   (training)
                                 z = μ          (inference)
                              │
                    ┌───────── Decoder ───────────┐
  z ∈ ℝ⁵  ──→  Linear(5, 16)  → BN → ReLU
             → Linear(16, 32) → BN → ReLU
             → Linear(32, 64) → BN → ReLU
             → Linear(64, 100)                  ← linear output
                    └──────────────────────────────┘
                              │
                              ▼
                         x̂ ∈ ℝ¹⁰⁰  (reconstructed profile)
```

### Parameter count

| Component | Parameters |
|-----------|-----------|
| Encoder backbone | 100×64 + 64×32 + 32×16 + biases + BN ≈ 9,600 |
| μ head | 16×5 + 5 = 85 |
| log σ² head | 16×5 + 5 = 85 |
| Decoder | 5×16 + 16×32 + 32×64 + 64×100 + biases + BN ≈ 9,600 |
| **Total** | **≈ 19,400** |

Lightweight by design — trainable on CPU if needed, fast on any GPU.

### Design decisions

1. **MLP, not CNN.**  The input is a 1D profile, not a 2D image.  Fully-connected
   layers with BatchNorm are the natural choice for tabular/vector data.  A CNN would
   impose translational equivariance that the profile does not possess (position 25%
   and position 75% have different biological meanings).

2. **LeakyReLU in encoder, ReLU in decoder.**  LeakyReLU prevents dead neurons in the
   encoder bottleneck; ReLU in the decoder produces clean positive activations for
   reconstruction.

3. **Dropout only in encoder.**  Dropout regularises the encoder, preventing it from
   memorising a trivial identity mapping.  The decoder is kept deterministic so
   reconstruction quality depends purely on the latent code quality.

4. **Linear decoder output (no activation).**  Input data is standardised (μ=0, σ=1
   per position).  A linear output can represent any real value and pairs naturally
   with MSE loss.  Sigmoid/tanh would constrain the output range and introduce
   saturation gradients.

5. **β-VAE with small β (0.001).**  The KL term `β · D_KL(q(z|x) ‖ p(z))` regularises
   the latent space towards 𝒩(0,I).  A small β prioritises reconstruction fidelity
   while still ensuring the latent space is smooth and compact—important when latents
   are used as quantitative traits where we want most variation captured in z, not
   discarded by an aggressive KL penalty.

6. **Cosine annealing LR schedule.**  Smoothly decays the learning rate over the full
   training duration, avoiding the need to tune step-decay milestones.

7. **Per-position standardisation.**  Each of the 100 profile positions is standardised
   independently (μ=0, σ=1).  This prevents positions with naturally larger variance
   (e.g., mid-body) from dominating the loss.

---

## Input

`rep_width_profiles.txt` — whitespace-separated, no header:

```
HN-23-11-CG-163  7.12  8.34  9.01  ...  3.45    ← 100 width values
SZ-21-09-CG-088  6.89  8.12  8.76  ...  3.21
...
```

Column 1 = plant identifier; columns 2–101 = median full-width profile values
(100 positions along the length-normalised axis, in mm-equivalent units after
calibration).  Produced by `pipeline/processing/representative_shape.py`.

---

## Output (under `runs/<name>/`)

| File | Description |
|------|-------------|
| `best.pt` | Checkpoint with lowest validation loss |
| `last_state_dict.pt` | Final model weights |
| `results.csv` | Per-epoch train/val loss, recon loss, KL loss |
| `latent_traits.csv` | **Main output** — `plant_id, latent_1..latent_5` |
| `recon_epoch_*.png` | 6-example reconstruction overlay (original vs recon) |
| `latent_pca.png` | 2D PCA projection of the 5-dim latent space |
| `loss_curve.png` | Train + validation loss over epochs |
| `args.json` | Run metadata for reproducibility |

### `latent_traits.csv` format

```csv
plant_id,latent_1,latent_2,latent_3,latent_4,latent_5
HN-23-11-CG-163,-0.342,1.207,-0.891,0.453,-0.112
SZ-21-09-CG-088,0.756,-0.623,0.334,-0.987,0.201
...
```

Each `latent_i` is a standard-normal-like value (μ≈0, σ≈1 across the population).
These five columns can be used directly as quantitative traits in GWAS software
(GAPIT, GEMMA, FarmCPU, etc.).

---

## Usage

```bash
cd vae/
python train_vae.py config.yaml
```

Override key hyperparameters via CLI:

```bash
python train_vae.py config.yaml --latent-dim 3 --epochs 200 --beta 0.01 --device cuda:0
```

---

## Interpreting the latent space

- **latent_pca.png** shows a 2D PCA of all 5 latent dimensions. Genotypes that are
  close together in this projection have similar width-profile shapes.
- Each `latent_i` captures a different axis of shape variation.  For example,
  `latent_1` might correlate with kernel elongation (length/width ratio), while
  `latent_2` might capture the position of maximum breadth (peak at 40% vs 60%).
- The biological interpretation of each latent dimension can be investigated by
  correlating `latent_i` with known morphological traits (length, width, area,
  circularity) from `final_output_plant_median.csv`, or by visualising the
  reconstructed profiles at extreme values of each latent dimension.

---

## When to retrain

- **New genetic population**: retrain if the new material differs substantially in
  kernel shape (e.g., popcorn vs dent vs flint).
- **Same population, more samples**: fine-tune from the existing checkpoint rather
  than training from scratch.
- **Different pipeline config**: if `num_width_samples` or calibration changes,
  retrain.
