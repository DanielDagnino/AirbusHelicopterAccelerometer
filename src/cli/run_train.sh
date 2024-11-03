#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

./cli/a1_train.py cfg/train-adam.yaml autoencoder_v1 0;
./cli/a1_train.py cfg/train-adam-var.yaml autoencoder_v1 0;
./cli/a1_train.py cfg/train-adam.yaml autoencoder_v1 0.1;
./cli/a1_train.py cfg/train-adam-var.yaml autoencoder_v1 0.1;
./cli/a1_train.py cfg/train-adam-w.yaml autoencoder_v1 0;
./cli/a1_train.py cfg/train-adam-var-w.yaml autoencoder_v1 0;
./cli/a1_train.py cfg/train-adam-w.yaml autoencoder_v1 0.1;
./cli/a1_train.py cfg/train-adam-var-w.yaml autoencoder_v1 0.1;
