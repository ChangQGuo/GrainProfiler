#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "running test"
python test_resnet_angle.py