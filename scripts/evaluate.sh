#!/bin/bash

MODEL=${1}
DATASET=${2}
SPARSITY=${3}
Q_FLOP_coef=${4}
Q_Para_coef=${5}


SPARSITY=$(printf "%.2f" "$SPARSITY")
Q_FLOP_coef=$(printf "%.2f" "$Q_FLOP_coef")
Q_Para_coef=$(printf "%.2f" "$Q_Para_coef")

LOG=log
PRETRAINED_MODEL_DIR=pretrained_model
COMPRESSED_MODEL_DIR=compressed_model
PRETRAINED_MODEL_PTH=${PRETRAINED_MODEL_DIR}/${MODEL}_${DATASET}_pretrained.pth
COMPRESSED_MODEL_PTH=${COMPRESSED_MODEL_DIR}/${MODEL}_${DATASET}_${SPARSITY}_${Q_FLOP_coef}_${Q_Para_coef}.pth


python -m evaluate --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --log_dir ${LOG}
