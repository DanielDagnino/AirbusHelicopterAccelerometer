#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

./cli/a1_train.py cfg/train-adam-w.yaml autoencoder_v1 0;
