#!/usr/bin/env bash

export CURRENT_DIR="."
export ROOT_DIR="/"

export TF_FORCE_GPU_ALLOW_GROWTH=true

cd $CURRENT_DIR

export PYTHONPATH="$CURRENT_DIR:$LIB_DIR:$PYTHONPATH"

echo "Application started"

python ./validation/comp_dgns/val_models/20230206_123655DGNs_without_D0_0/generators.py

echo " Application finished"