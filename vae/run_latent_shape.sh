#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running latent shape semantics analysis"
python latent_shape_explorer.py runs_new/profile_vae_latent5
