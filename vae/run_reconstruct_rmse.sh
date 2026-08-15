#!/bin/bash

# Path to rep_width_profiles.txt — MUST be the same file (same row order) that
# was used for VAE training, otherwise the test split will not match the paper.
PROFILE_PATH="../HaiNan_results_100images/rep_width_profiles.txt"

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running per-test-sample reconstruction RMSE (boxplot + table)"
python reconstruct_test_rmse.py runs_new/profile_vae_latent5 --profile "$PROFILE_PATH"
