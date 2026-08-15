"""
vae/train_vae.py — Train 1D-CNN β-VAE on width profiles, export latent traits.

Output (under runs/<run_name>/):
  best.pt                 checkpoint (lowest val loss)
  results.csv             per-epoch train/val loss
  latent_traits.csv       ALL samples: plant_id + latent_1..latent_5  ← GWAS input
  recon_epoch_*.png       validation reconstruction overlay (every 50 epochs)
  test_recon_grid.png     9-sample test-set reconstruction (literature figure)
  test_metrics.json       test loss metrics
  latent_pca.png          2D PCA of latent space (all samples)
  loss_curve.png          train + val loss over epochs
  args.json               run metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import WidthProfileDataset, split_dataset
from model import ProfileVAE, ProfileVAEConfig, vae_loss

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

RECON_BLUE = '#2166AC'
RECON_ORANGE = '#D8A03D'


def save_recon_grid(originals, recons, labels, output_path, title, denorm_fn):
    """3×3 reconstruction overlay grid — blue=original, orange=reconstructed."""
    if not _HAS_MPL:
        return
    originals = denorm_fn(originals)
    recons = denorm_fn(recons)
    n = min(9, len(originals))
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.5 * rows), facecolor='white')
    axes = axes.flatten() if n > 1 else [axes]
    for i in range(n):
        ax = axes[i]
        x = np.arange(1, len(originals[i]) + 1)
        ax.plot(x, originals[i], color=RECON_BLUE, linewidth=1.5, label='Original')
        ax.plot(x, recons[i], color=RECON_ORANGE, linewidth=1.5, linestyle='--',
                label='Reconstructed')
        ax.fill_between(x, originals[i], recons[i], alpha=0.10, color='#888888')
        ax.set_title(labels[i], fontsize=9)
        ax.set_xlabel('Normalized position along morphological axis (%)')
        ax.set_ylabel('Full-width (scaled)')
        ax.legend(fontsize=7, frameon=False, loc='upper right')
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_latent_pca_plot(latent_vectors, labels, output_path):
    """2D PCA of latent space."""
    if not _HAS_MPL:
        return
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return
    coords = PCA(n_components=2).fit_transform(latent_vectors)
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor='white')
    ax.scatter(coords[:, 0], coords[:, 1],
               c=coords[:, 0], cmap='RdBu_r', alpha=0.75,
               edgecolors='#444444', linewidth=0.3, s=50)
    ax.set_xlabel('Latent PC1')
    ax.set_ylabel('Latent PC2')
    ax.set_title('PCA of 5-dim VAE Latent Space')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.7)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.7)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_loss_curve(results_csv_path, output_path):
    """Train + val loss."""
    if not _HAS_MPL:
        return
    epochs, train, val = [], [], []
    with open(results_csv_path, 'r') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train.append(float(row['loss']))
            if 'val_loss' in row and row['val_loss']:
                val.append(float(row['val_loss']))
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='white')
    ax.plot(epochs, train, color=RECON_BLUE, linewidth=1.5, label='Train loss')
    if val and len(val) == len(epochs):
        ax.plot(epochs, val, color='#B2182B', linewidth=1.5, label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('VAE Training Curve')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    text = str(device_arg or 'auto').lower()
    if text == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if text.startswith('cuda') and not torch.cuda.is_available():
        print(f'WARNING: requested {device_arg}, CUDA unavailable; using CPU.')
        return torch.device('cpu')
    return torch.device(device_arg)


_FIELDS = ['epoch', 'loss', 'recon_loss', 'kl_loss',
           'val_loss', 'val_recon_loss', 'val_kl_loss']


def write_results_header(path: Path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=_FIELDS).writeheader()


def append_results(path: Path, row: dict):
    with open(path, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)


def export_latents(model, loader, device, output_csv: Path, latent_dim: int):
    """Export plant_id + latent_1..latent_k for all samples."""
    model.eval()
    fieldnames = ['plant_id'] + [f'latent_{i + 1}' for i in range(latent_dim)]
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch['profile'].to(device)
            mu, _ = model.encode(x)
            for label, vec in zip(batch['label'], mu.detach().cpu().numpy()):
                row = {'plant_id': label}
                for idx, val in enumerate(vec):
                    row[f'latent_{idx + 1}'] = float(val)
                rows.append(row)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, loader, device, beta: float):
    """Return dict of average loss/recon/kl over a data loader."""
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['profile'].to(device)
            recon, mu, logvar, _ = model(x)
            loss, r, k = vae_loss(recon, x, mu, logvar, beta=beta)
            total_loss += float(loss.item()) * x.shape[0]
            total_recon += float(r.item()) * x.shape[0]
            total_kl += float(k.item()) * x.shape[0]
            n += x.shape[0]
    return {
        'samples': n,
        'loss': round(total_loss / n, 6),
        'recon_loss': round(total_recon / n, 6),
        'kl_loss': round(total_kl / n, 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Train β-VAE on kernel width profiles.')
    p.add_argument('config', nargs='?', default='config.yaml')
    p.add_argument('--latent-dim', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--beta', type=float, default=None)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config_dir = config_path.parent

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    output_cfg = config.get('output', {})

    latent_dim = int(args.latent_dim or model_cfg.get('latent_dim', 5))
    epochs = int(args.epochs or train_cfg.get('epochs', 300))
    batch_size = int(args.batch_size or train_cfg.get('batch_size', 32))
    beta = float(args.beta or train_cfg.get('beta', 0.001))
    device = resolve_device(args.device or train_cfg.get('device', 'auto'))

    early_stop = int(train_cfg.get('early_stop_patience', 30))
    lr_patience = int(train_cfg.get('lr_patience', 25))

    set_seed(int(train_cfg.get('seed', 42)))

    # ---- Data: train 70% / val 20% / test 10% ----
    profile_path = Path(str(data_cfg.get('profile_path', 'rep_width_profiles.txt')))
    if not profile_path.is_absolute():
        profile_path = (config_dir / profile_path).resolve()

    full_dataset = WidthProfileDataset(profile_path)
    train_ds, val_ds, test_ds = split_dataset(full_dataset)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=int(train_cfg.get('num_workers', 0)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f'Loaded {full_dataset.num_samples} profiles ({full_dataset.input_dim}-dim)')
    print(f'  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}')

    # ---- Model ----
    vae_config = ProfileVAEConfig(
        input_dim=full_dataset.input_dim,
        latent_dim=latent_dim,
        channels=[int(c) for c in model_cfg.get('channels', [32, 64])],
        kernel_size=int(model_cfg.get('kernel_size', 5)),
        dropout=float(model_cfg.get('dropout', 0.1)),
    )
    model = ProfileVAE(vae_config).to(device)
    print(f'Model: 1D-CNN β-VAE, {sum(p.numel() for p in model.parameters()):,} params')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get('lr', 0.0005)),
        weight_decay=float(train_cfg.get('weight_decay', 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=1e-6,
    )

    # ---- Output dir ----
    project_dir = config_dir / str(output_cfg.get('project', 'runs'))
    run_dir = project_dir / str(output_cfg.get('name', 'profile_vae_latent5'))
    run_dir.mkdir(parents=True, exist_ok=bool(output_cfg.get('exist_ok', True)))
    results_csv = run_dir / 'results.csv'
    write_results_header(results_csv)

    best_val_loss = float('inf')
    best_epoch = 0
    stall = 0
    val_fixed = next(iter(DataLoader(val_ds, batch_size=9, shuffle=True)))

    print(f'\nTraining on {device} — max {epochs} epochs, β={beta}, batch={batch_size}')
    print(f'Early stop patience: {early_stop}  |  LR patience: {lr_patience}')
    print(f'Output → {run_dir}\n')

    # ===================== Training loop =====================
    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        t_loss = t_recon = t_kl = 0.0
        nb = 0
        for batch in train_loader:
            x = batch['profile'].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar, _ = model(x)
            loss, r, k = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            t_loss += float(loss.item()); t_recon += float(r.item()); t_kl += float(k.item())
            nb += 1
        t_loss /= max(nb, 1); t_recon /= max(nb, 1); t_kl /= max(nb, 1)

        # -- Validate --
        model.eval()
        v_loss = v_recon = v_kl = 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['profile'].to(device)
                recon, mu, logvar, _ = model(x)
                loss, r, k = vae_loss(recon, x, mu, logvar, beta=beta)
                v_loss += float(loss.item()); v_recon += float(r.item()); v_kl += float(k.item())
                nv += 1
        v_loss /= max(nv, 1); v_recon /= max(nv, 1); v_kl /= max(nv, 1)

        scheduler.step(v_loss)

        append_results(results_csv, {
            'epoch': epoch,
            'loss': round(t_loss, 6), 'recon_loss': round(t_recon, 6), 'kl_loss': round(t_kl, 6),
            'val_loss': round(v_loss, 6), 'val_recon_loss': round(v_recon, 6), 'val_kl_loss': round(v_kl, 6),
        })

        # Early stopping + checkpoint
        if v_loss < best_val_loss:
            best_val_loss = v_loss; best_epoch = epoch; stall = 0
            torch.save({
                'model_state': model.state_dict(),
                'config': vae_config,
                'col_mean': full_dataset.col_mean,
                'col_std': full_dataset.col_std,
                'beta': beta, 'epoch': epoch,
            }, run_dir / 'best.pt')
        else:
            stall += 1

        # Reconstruction plots every 50 epochs + first + last
        if epoch in (1, epochs) or epoch % 50 == 0 or stall == 0:
            xv = val_fixed['profile'].to(device)
            with torch.no_grad():
                rv, _, _, _ = model(xv)
            save_recon_grid(
                xv.cpu().numpy(), rv.cpu().numpy(), val_fixed['label'],
                run_dir / f'recon_epoch_{epoch:04d}.png',
                f'Validation Reconstructions — Epoch {epoch}',
                full_dataset.denormalize,
            )
            print(f'  epoch {epoch:04d}/{epochs}:  '
                  f'train={t_loss:.6f} (r={t_recon:.6f} kl={t_kl:.6f})  '
                  f'val={v_loss:.6f} (r={v_recon:.6f} kl={v_kl:.6f})  '
                  f'lr={optimizer.param_groups[0]["lr"]:.2e}')

        if stall >= early_stop:
            print(f'\n  Early stop at epoch {epoch} — best val_loss={best_val_loss:.6f} '
                  f'at epoch {best_epoch}')
            break

    # ===================== Test evaluation =====================
    print(f'\n{"="*50}')
    print('Loading best model for test evaluation...')
    ckpt = torch.load(run_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)

    test_metrics = evaluate(model, test_loader, device, beta)
    print(f'Test set ({test_metrics["samples"]} samples): '
          f'loss={test_metrics["loss"]:.6f}  '
          f'recon={test_metrics["recon_loss"]:.6f}  '
          f'kl={test_metrics["kl_loss"]:.6f}')

    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # 9-sample test reconstruction for literature
    test_fixed = next(iter(DataLoader(test_ds, batch_size=9, shuffle=True)))
    xt = test_fixed['profile'].to(device)
    with torch.no_grad():
        rt, _, _, _ = model(xt)
    save_recon_grid(
        xt.cpu().numpy(), rt.cpu().numpy(), test_fixed['label'],
        run_dir / 'test_recon_grid.png',
        'Test Set Reconstructions',
        full_dataset.denormalize,
    )
    print(f'Test reconstruction figure saved → {run_dir / "test_recon_grid.png"}')

    # ===================== Final exports =====================
    # Latent traits for ALL samples (train+val+test)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    export_latents(model, full_loader, device,
                   run_dir / 'latent_traits.csv', latent_dim)
    print(f'Latent traits (all {full_dataset.num_samples} samples) → {run_dir / "latent_traits.csv"}')

    # Latent PCA
    model.eval()
    all_mu = []
    with torch.no_grad():
        for batch in full_loader:
            mu, _ = model.encode(batch['profile'].to(device))
            all_mu.append(mu.cpu().numpy())
    save_latent_pca_plot(np.concatenate(all_mu, axis=0),
                         full_dataset.labels,
                         run_dir / 'latent_pca.png')

    save_loss_curve(results_csv, run_dir / 'loss_curve.png')

    with open(run_dir / 'args.json', 'w', encoding='utf-8') as f:
        json.dump({
            'profile_path': str(profile_path),
            'input_dim': full_dataset.input_dim,
            'latent_dim': latent_dim,
            'channels': vae_config.channels,
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'test_samples': len(test_ds),
            'epochs_completed': epoch,
            'best_epoch': best_epoch,
            'batch_size': batch_size,
            'beta': beta,
            'device': str(device),
        }, f, indent=2)

    print(f'\nDone. Best epoch: {best_epoch}  |  '
          f'Test loss: {test_metrics["loss"]:.6f}')
    print(f'  Latent traits → {run_dir / "latent_traits.csv"}')
    print(f'  Test figure   → {run_dir / "test_recon_grid.png"}')


if __name__ == '__main__':
    main()
