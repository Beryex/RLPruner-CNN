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


# Step 1: train model (This is optional, skip this step if you have pretrained model)
# If you skip shis, make sure your pretrained model is named as "${model}_${dataset}_original.pth"
python -m train --model ${MODEL} --dataset ${DATASET} --device cuda \
                --output_dir ${PRETRAINED_MODEL_DIR} \
                --log_dir ${LOG} --use_wandb


# Step 2: Compress trained model
python -m compress --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --sparsity ${SPARSITY} --prune_strategy ${prune_strategy} --ppo \
                   --pretrained_dir ${PRETRAINED_MODEL_DIR} \
                   --compressed_dir ${COMPRESSED_MODEL_DIR} \
                   --checkpoint_dir ${CKPT_DIR} \
                   --log_dir ${LOG} --use_wandb \
                   # --resume --resume_epoch 5


# Step 3: Evaluate the compression results
python -m evaluate --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --log_dir ${LOG}
