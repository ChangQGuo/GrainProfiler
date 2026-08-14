#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running population-median latent perturbation grid"
python latent_perturbation_grid.py runs/profile_vae_latent5
