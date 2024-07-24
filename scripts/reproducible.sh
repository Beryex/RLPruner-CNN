#!/bin/bash


MODEL=vgg16
DATASET=cifar100

LOG=log
CKPT=checkpoint
MODEL_DIR=models
PRETRAINED_MODEL_DIR=pretrained_model
COMPRESSED_MODEL_DIR=compressed_model


if [ ! -d "$CKPT" ]; then
    mkdir -p "$CKPT"
fi

if [ ! -d "$MODEL_DIR" ]; then
    mkdir "$MODEL_DIR"
fi

if [ ! -d "$PRETRAINED_MODEL_DIR" ]; then
    mkdir -p "$PRETRAINED_MODEL_DIR"
fi

if [ ! -d "$COMPRESSED_MODEL_DIR" ]; then
    mkdir -p "$COMPRESSED_MODEL_DIR"
fi

if [ ! -d "$LOG" ]; then
    mkdir -p "$LOG"
fi


# Step 1: train model (This is optional, skip this step if you have pretrained model)
CUDA_VISIBLE_DEVICES=0 python -m train.py -m $MODEL -ds $DATASET -dev cuda --use-wandb -rs 1

# Step 2: Compress trained model
CUDA_VISIBLE_DEVICES=0 python -m compress.py -m $MODEL -ds $DATASET -dev cuda --use-wandb -rs 1
