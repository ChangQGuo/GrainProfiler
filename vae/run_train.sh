#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running train"
python train_vae.py config.yaml