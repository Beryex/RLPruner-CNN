#!/bin/bash

MODEL=vgg16
DATASET=cifar100

SPARSITY=0.60

LOG=log
CKPT=checkpoint
PRETRAINED_MODEL_DIR=pretrained_model
COMPRESSED_MODEL_DIR=compressed_model
CKPT_DIR=${CKPT}/${MODEL}_${DATASET}
PRETRAINED_MODEL_PTH=${PRETRAINED_MODEL_DIR}/${MODEL}_${DATASET}_original.pth
COMPRESSED_MODEL_PTH=${COMPRESSED_MODEL_DIR}/${MODEL}_${DATASET}_${SPARSITY}.pth

# RL searching and updating: 
# bash scripts/hyperparameter_search.sh "noise_var,ppo_clip,step_length" "0.02;0.04;0.06,0.15;0.25;0.35,0.5;0.75;1"
# RL learning: 
# bash scripts/hyperparameter_search.sh "sample_num,sample_step,lr_epoch" "5;10;15,1;2,5;10;15"
# fine tuning: 
# bash scripts/hyperparameter_search.sh "stu_co,KD_temperature,lr,fine_tune_epoch,warmup_epoch" "0;0.3;0.6;0.9,0.5;1;2,5e-2;1e-2;5e-3,10;15;20,2;5;8"
SUPPORT_HYPERPARAMETER=("noise_var" "ppo_clip" "step_length" \
                        "sample_num" "sample_step" "lr_epoch" \
                        "greedy_epsilon" "prune_strategy" \
                        "stu_co" "KD_temperature" "lr" "fine_tune_epoch" "warmup_epoch")

hyparameter_list="${1}"
hyparameter_values="${2}"

IFS=',' read -r -a hyparameter_list_array <<< "$hyparameter_list"
IFS=',' read -r -a hyparameter_values_array <<< "$hyparameter_values"

for parameter in "${hyparameter_list_array[@]}"; do
    if [[ ! " ${SUPPORT_HYPERPARAMETER[@]} " =~ " ${parameter} " ]]; then
        echo "Error: Unsupported hyperparameter ${parameter}"
        exit 1
    fi
done

declare -A hyparameter_values_map
for i in "${!hyparameter_list_array[@]}"; do
    IFS=';' read -r -a values <<< "${hyparameter_values_array[$i]}"
    hyparameter_values_map["${hyparameter_list_array[$i]}"]="${values[*]}"
done

function enumerate {
    local index=$1
    local -n current_comb=$2

    if [ $index -eq ${#hyparameter_list_array[@]} ]; then
        local cmd="python -m compress --model ${MODEL} --dataset ${DATASET} --device cuda \
                   --sparsity ${SPARSITY} --ppo \
                   --pretrained_pth ${PRETRAINED_MODEL_PTH} \
                   --compressed_pth ${COMPRESSED_MODEL_PTH} \
                   --checkpoint_dir ${CKPT_DIR} \
                   --log_dir ${LOG} --use_wandb --random_seed 1"

        for param in "${!current_comb[@]}"; do
            cmd+=" --${param} ${current_comb[$param]}"
        done

        echo "Running: $cmd"
        eval $cmd
        return
    fi

    local param=${hyparameter_list_array[$index]}
    IFS=' ' read -r -a values <<< "${hyparameter_values_map[$param]}"

    for value in "${values[@]}"; do
        current_comb["$param"]="$value"
        enumerate $((index + 1)) current_comb
    done
}

declare -A current_comb
enumerate 0 current_comb
