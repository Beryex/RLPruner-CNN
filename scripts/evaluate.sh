#!/bin/bash

MODEL=${1}
DATASET=${2}

SPARSITY=${3}

LOG=log
PRETRAINED_MODEL_DIR=pretrained_model
COMPRESSED_MODEL_DIR=compressed_model
PRETRAINED_MODEL_PTH=${PRETRAINED_MODEL_DIR}/${MODEL}_${DATASET}_original.pth
COMPRESSED_MODEL_PTH=${COMPRESSED_MODEL_DIR}/${MODEL}_${DATASET}_${SPARSITY}.pth


if [ ! -d "${LOG}" ]; then
    mkdir -p "${LOG}"
fi

if [ ! -d "${PRETRAINED_MODEL_DIR}" ]; then
    mkdir -p "${PRETRAINED_MODEL_DIR}"
fi

if [ ! -d "${COMPRESSED_MODEL_DIR}" ]; then
    mkdir -p "${COMPRESSED_MODEL_DIR}"
fi


python -m evaluate --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --log_dir ${LOG}
