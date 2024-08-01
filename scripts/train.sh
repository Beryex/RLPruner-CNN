#!/bin/bash

MODEL=${1}
DATASET=${2}

LOG=log
PRETRAINED_MODEL_DIR=pretrained_model
PRETRAINED_MODEL_PTH=${PRETRAINED_MODEL_DIR}/${MODEL}_${DATASET}_original.pth


if [ ! -d "${LOG}" ]; then
    mkdir -p "${LOG}"
fi

if [ ! -d "${PRETRAINED_MODEL_DIR}" ]; then
    mkdir -p "${PRETRAINED_MODEL_DIR}"
fi


python -m train --model ${MODEL} --dataset ${DATASET} --device cuda \
                --output_pth ${PRETRAINED_MODEL_PTH} \
                --log_dir ${LOG} --use_wandb
