#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running VAE sample reconstruction (4 representative samples)"
python reconstruct_samples.py runs_new/profile_vae_latent5
