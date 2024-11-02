#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

./cli/a2_eval.py cfg/valid.yaml autoencoder_v1 64
