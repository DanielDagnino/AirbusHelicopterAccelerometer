#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

#./cli/a1_train.py cfg/train-adam.yaml autoencoder_v1 0;
#./cli/a1_train.py cfg/train-adam-var.yaml autoencoder_v1 0;
#
#./cli/a1_train.py cfg/train-adam-norm.yaml autoencoder_v1 0;
#./cli/a1_train.py cfg/train-adam-var-norm.yaml autoencoder_v1 0;
#
#./cli/a1_train.py cfg/train-adam-var.yaml autoencoder_v1 0.1;
./cli/a1_train.py cfg/train-adam.yaml autoencoder_v1 0.1;
./cli/a1_train.py cfg/train-adam.yaml autoencoder_v1 0.5;

#
#./cli/a1_train.py cfg/train-adam-var-norm.yaml autoencoder_v1 0.1;
#./cli/a1_train.py cfg/train-adam-norm.yaml autoencoder_v1 0.1;
#