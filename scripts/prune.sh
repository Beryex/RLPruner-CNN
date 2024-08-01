#!/bin/bash

MODEL=${1}
DATASET=${2}

SPARSITY=${3}

LOG=log
CKPT=checkpoint
PRETRAINED_MODEL_DIR=pretrained_model
COMPRESSED_MODEL_DIR=compressed_model
CKPT_DIR=${CKPT}/${MODEL}_${DATASET}
PRETRAINED_MODEL_PTH=${PRETRAINED_MODEL_DIR}/${MODEL}_${DATASET}_original.pth
COMPRESSED_MODEL_PTH=${COMPRESSED_MODEL_DIR}/${MODEL}_${DATASET}_${SPARSITY}.pth


if [ ! -d "${LOG}" ]; then
    mkdir -p "${LOG}"
fi

if [ ! -d "${CKPT}" ]; then
    mkdir -p "${CKPT}"
fi

if [ ! -d "${PRETRAINED_MODEL_DIR}" ]; then
    mkdir -p "${PRETRAINED_MODEL_DIR}"
fi

if [ ! -d "${COMPRESSED_MODEL_DIR}" ]; then
    mkdir -p "${COMPRESSED_MODEL_DIR}"
fi


python -m compress --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --sparsity ${SPARSITY} --prune_strategy variance \
                   --greedy_epsilon 0 --ppo \
                   --noise_var 0.04 --ppo_clip 0.2 \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --checkpoint_dir ${CKPT_DIR} \
                   --log_dir ${LOG}  --use_wandb \
                   # --resume --resume_epoch 5
