#!/bin/bash

MODEL=vgg16
DATASET=cifar100

SPARSITY=0.60

LOG=log
CKPT=checkpoint
MODEL_DIR=models
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

if [ ! -d "${MODEL_DIR}" ]; then
    mkdir "${MODEL_DIR}"
fi

if [ ! -d "${PRETRAINED_MODEL_DIR}" ]; then
    mkdir -p "${PRETRAINED_MODEL_DIR}"
fi

if [ ! -d "${COMPRESSED_MODEL_DIR}" ]; then
    mkdir -p "${COMPRESSED_MODEL_DIR}"
fi


# Step 1: train model (This is optional, skip this step if you have pretrained model)
# If you skip shis, make sure your pretrained model is named as "${model}_${dataset}_original.pth"
python -m train --model ${MODEL} --dataset ${DATASET} --device cuda \
                --model_dir ${MODEL_DIR} --output_pth ${PRETRAINED_MODEL_PTH} \
                --log_dir ${LOG} --use_wandb



# Step 2: Compress trained model
python -m compress --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --sparsity ${SPARSITY} --prune_strategy variance \
                   --greedy_epsilon 0 --ppo \
                   --noise_var 0.04 --ppo_clip 0.2 \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --checkpoint_dir ${CKPT_DIR} \
                   --log_dir ${LOG} --use_wandb


# Step 3: Evaluate the compression results
python -m evaluate --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --log_dir ${LOG}
