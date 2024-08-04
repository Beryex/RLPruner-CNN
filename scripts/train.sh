#!/bin/bash

MODEL=${1}
DATASET=${2}

LOG=log
PRETRAINED_MODEL_DIR=pretrained_model
PRETRAINED_MODEL_PTH=${PRETRAINED_MODEL_DIR}/${MODEL}_${DATASET}_original.pth


python -m train --model ${MODEL} --dataset ${DATASET} --device cuda \
                --output_dir ${PRETRAINED_MODEL_DIR} \
                --log_dir ${LOG} --use_wandb
