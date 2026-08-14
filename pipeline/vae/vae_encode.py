"""
Stage 6 — Encode 100-dim width profiles to 5-dim latent traits.

READS:
  - rep_width_profiles.txt        (Stage 5 output)
  - vae_checkpoint.pt             (trained VAE model)

WRITES:
  - latent_traits.csv             (plant_id, latent_1..latent_5)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config


def run(config_path: str):
    config = load_config(config_path)
    output_dir = config['output']['base_dir']

    # --- resolve paths ---
    profile_txt = os.path.join(
        output_dir,
        config['output'].get('rep_width_profiles_txt', 'rep_width_profiles.txt'),
    )
    latent_csv = os.path.join(
        output_dir,
        config['output'].get('latent_traits_csv', 'latent_traits.csv'),
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(script_dir)
    checkpoint_cfg = config.get('models', {}).get('vae_checkpoint', 'vae/vae_checkpoint.pt')
    if os.path.isabs(checkpoint_cfg):
        checkpoint_path = checkpoint_cfg
    else:
        checkpoint_path = os.path.join(pipeline_dir, checkpoint_cfg)

    # --- validate inputs ---
    if not os.path.exists(profile_txt):
        print(f'ERROR: {profile_txt} not found. Run Stage 5 (shapes) first.')
        sys.exit(1)
    if not os.path.exists(checkpoint_path):
        print(f'ERROR: {checkpoint_path} not found. '
              f'Set models.vae_checkpoint in config.yaml (relative to pipeline dir or absolute).')
        sys.exit(1)

    # --- load checkpoint ---
    sys.path.insert(0, script_dir)
    from model import Encoder, ProfileVAEConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'VAE Encode — device: {device}')

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Convert normalization stats to numpy robustly.  Depending on how the
    # checkpoint was produced, col_mean/col_std may already be numpy arrays
    # (train_vae.py pickles dataset.col_mean directly) or torch tensors.
    col_mean = ckpt['col_mean']
    col_std = ckpt['col_std']
    if hasattr(col_mean, 'cpu'):
        col_mean = col_mean.cpu().numpy()
    if hasattr(col_std, 'cpu'):
        col_std = col_std.cpu().numpy()
    col_mean = np.asarray(col_mean, dtype=np.float32)
    col_std = np.asarray(col_std, dtype=np.float32)
    col_std[col_std < 1e-8] = 1.0

    vae_config: ProfileVAEConfig = ckpt['config']
    encoder = Encoder(vae_config)
    enc_state = {k[8:]: v for k, v in ckpt['model_state'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(enc_state)
    encoder.to(device)
    encoder.eval()
    print(f'Loaded VAE checkpoint (epoch {ckpt["epoch"]}, beta={ckpt["beta"]})')

    # --- read profiles ---
    print(f'Reading: {profile_txt}')
    names, rows = [], []
    with open(profile_txt, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            names.append(parts[0])
            rows.append([float(v) for v in parts[1:]])

    if len(names) < 1:
        print('ERROR: No valid profiles found.')
        sys.exit(1)

    X = np.array(rows, dtype=np.float32)
    print(f'  {len(names)} plants × {X.shape[1]} width positions')

    # --- normalize (same as training) ---
    X_norm = (X - col_mean) / col_std

    # --- encode in batches ---
    batch_size = 256
    all_latents = []
    with torch.no_grad():
        for start in range(0, len(X_norm), batch_size):
            batch = torch.from_numpy(X_norm[start:start + batch_size]).to(device)
            mu, _ = encoder(batch)  # use mu (not sample) for deterministic encoding
            all_latents.append(mu.cpu().numpy())

    Z = np.concatenate(all_latents, axis=0)
    print(f'  Encoded → {Z.shape[1]}-dim latent vectors')

    # --- write output ---
    cols = ['plant_id'] + [f'latent_{i+1}' for i in range(Z.shape[1])]
    df = pd.DataFrame(np.column_stack([names, Z]), columns=cols)
    for c in cols[1:]:
        df[c] = df[c].astype(float)
    df.to_csv(latent_csv, index=False, float_format='%.6f')
    print(f'Wrote: {latent_csv}  ({len(df)} rows)')
    print('Done.')


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    run(config_path)
