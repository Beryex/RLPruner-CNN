#!/bin/bash

MODEL=${1}
DATASET=${2}
SPARSITY=${3}
prune_strategy=${4}

LOG=log
CKPT=checkpoint
PRETRAINED_MODEL_DIR=pretrained_model
COMPRESSED_MODEL_DIR=compressed_model
CKPT_DIR=${CKPT}/${MODEL}_${DATASET}
PRETRAINED_MODEL_PTH=${PRETRAINED_MODEL_DIR}/${MODEL}_${DATASET}_original.pth
COMPRESSED_MODEL_PTH=${COMPRESSED_MODEL_DIR}/${MODEL}_${DATASET}_${SPARSITY}.pth


python -m compress --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --sparsity ${SPARSITY} --prune_strategy ${prune_strategy} --ppo \
                   --pretrained_dir ${PRETRAINED_MODEL_DIR} \
                   --compressed_dir ${COMPRESSED_MODEL_DIR} \
                   --checkpoint_dir ${CKPT_DIR} \
                   --log_dir ${LOG} --use_wandb \
                   # --resume --resume_epoch 5

python -m evaluate --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --log_dir ${LOG}
