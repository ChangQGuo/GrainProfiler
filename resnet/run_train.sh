#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running train"
python train_resnet_angle.py config.yaml