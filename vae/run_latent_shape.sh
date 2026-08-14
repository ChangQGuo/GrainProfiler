#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running latent shape semantics analysis"
python latent_shape_explorer.py runs/profile_vae_latent5
