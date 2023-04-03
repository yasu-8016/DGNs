#!/usr/bin/env bash

export CURRENT_DIR="."
export ROOT_DIR="/"

export TF_FORCE_GPU_ALLOW_GROWTH=true

cd $CURRENT_DIR

export PYTHONPATH="$CURRENT_DIR:$LIB_DIR:$PYTHONPATH"

echo "Application started"

python ./biz/ga_main.py

echo " Application finished"