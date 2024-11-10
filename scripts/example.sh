#!/bin/bash

MODEL=vgg19
DATASET=cifar100
SPARSITY=0.95
Q_FLOP_coef=0.00
Q_Para_coef=0.00

SPARSITY=$(printf "%.2f" "$SPARSITY")
Q_FLOP_coef=$(printf "%.2f" "$Q_FLOP_coef")
Q_Para_coef=$(printf "%.2f" "$Q_Para_coef")

LOG=log
CKPT=checkpoint
PRETRAINED_MODEL_DIR=pretrained_model
COMPRESSED_MODEL_DIR=compressed_model
CKPT_DIR=${CKPT}/${MODEL}_${DATASET}_${SPARSITY}_${Q_FLOP_coef}_${Q_Para_coef}
PRETRAINED_MODEL_PTH=${PRETRAINED_MODEL_DIR}/${MODEL}_${DATASET}_pretrained.pth
COMPRESSED_MODEL_PTH=${COMPRESSED_MODEL_DIR}/${MODEL}_${DATASET}_${SPARSITY}_${Q_FLOP_coef}_${Q_Para_coef}.pth


# Step 1: train model (This is optional, skip this step if you have pretrained model)
# If you skip shis, make sure your pretrained model is named as "${model}_${dataset}_pretrained.pth"
python -m train --model ${MODEL} --dataset ${DATASET} --device cuda \
                --output_dir ${PRETRAINED_MODEL_DIR} \
                --log_dir ${LOG} --use_wandb


# Step 2: Compress trained model
python -m compress --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --sparsity ${SPARSITY} --prune_strategy taylor --ppo \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_dir ${COMPRESSED_MODEL_DIR} \
                   --checkpoint_dir ${CKPT_DIR} \
                   --log_dir ${LOG} --use_wandb --save_model \
                   # --resume --resume_epoch 5


# Step 3: Evaluate the compression results
python -m evaluate --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --log_dir ${LOG}
