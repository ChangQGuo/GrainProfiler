#!/bin/bash

# Path to rep_width_profiles.txt (edit this to match your data location)
PROFILE_PATH="../HaiNan_results_100images/rep_width_profiles.txt"

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running latent interpretability analysis"
python interpret_latents.py runs/profile_vae_latent5 --profile "$PROFILE_PATH"
