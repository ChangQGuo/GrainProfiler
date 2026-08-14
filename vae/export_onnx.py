"""Export trained VAE decoder to ONNX for GUI inference (no PyTorch needed at runtime)."""

import os
import sys

import numpy as np
import torch

# Use the VAE model definition from the vae/ directory
from model import Decoder, ProfileVAEConfig

def main():
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'runs_new', 'profile_vae_latent5', 'best.pt',
    )
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'onnx_models',
    )
    os.makedirs(out_dir, exist_ok=True)

    # --- Load checkpoint ---
    print(f'Loading: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract decoder weights
    dec_state = {}
    for k, v in ckpt['model_state'].items():
        if k.startswith('decoder.'):
            dec_state[k[8:]] = v  # strip 'decoder.' prefix

    config: ProfileVAEConfig = ckpt['config']
    decoder = Decoder(config)
    decoder.load_state_dict(dec_state)
    decoder.eval()
    print(f'Decoder loaded ({sum(p.numel() for p in decoder.parameters())} params)')

    # --- Export to ONNX ---
    dummy = torch.randn(1, config.latent_dim)  # (1, 5)
    onnx_path = os.path.join(out_dir, 'profile_vae_latent5.onnx')

    torch.onnx.export(
        decoder,
        dummy,
        onnx_path,
        input_names=['latent'],
        output_names=['profile'],
        dynamic_axes={
            'latent': {0: 'batch'},
            'profile': {0: 'batch'},
        },
        opset_version=17,
    )
    print(f'Exported ONNX: {onnx_path}')

    # --- Save normalization params ---
    col_mean = ckpt['col_mean']  # (1, 100)
    col_std = ckpt['col_std']    # (1, 100)
    np.save(os.path.join(out_dir, 'vae_col_mean.npy'), col_mean)
    np.save(os.path.join(out_dir, 'vae_col_std.npy'), col_std)
    print(f'Saved normalization params to {out_dir}/')

    # --- Quick verification ---
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    test_out = session.run(None, {'latent': dummy.numpy().astype(np.float32)})[0]
    # PyTorch reference
    with torch.no_grad():
        ref = decoder(dummy).numpy()
    diff = np.abs(test_out - ref).max()
    print(f'Verification: ONNX vs PyTorch max diff = {diff:.2e} (should be < 1e-5)')
    assert diff < 1e-4, f'ONNX export verification failed! diff={diff}'
    print('ONNX export OK.')


if __name__ == '__main__':
    main()
