#!/bin/bash

MODEL=vgg16
DATASET=cifar100

LOG=log
CKPT=checkpoint
MODEL_DIR=models
PRETRAINED_MODEL_DIR=pretrained_model
COMPRESSED_MODEL_DIR=compressed_model


if [ ! -d "$LOG" ]; then
    mkdir -p "$LOG"
fi

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


# Step 1: train model (This is optional, skip this step if you have pretrained model)
# If you skip shis, make sure your pretrained model is named as "${model}_${dataset}_original.pth"
CUDA_VISIBLE_DEVICES=0 python -m train --model $MODEL --dataset $DATASET --device cuda \
                                       --model_dir $MODEL_DIR --output $PRETRAINED_MODEL_DIR \
                                       --log_dir $LOG --use_wandb -rs 1721847455

# Step 2: Compress trained model
CUDA_VISIBLE_DEVICES=0 python -m compress --model $MODEL --dataset $DATASET --device cuda \
                                          --sparsity 0.95 --greedy_epsilon 0 --ppo \
                                          --pretrained_dir $PRETRAINED_MODEL_DIR \
                                          --compressed_dir $COMPRESSED_MODEL_DIR \
                                          --checkpoint_dir $CKPT \
                                          --log_dir $LOG --use_wandb -rs 1
