import subprocess
import itertools
import sys
import argparse


# Usage: python hyperparameter_search.py -hp "stu_co,KD_temperature,lr,post_training_epoch,warmup_epoch" \
#                                        -v "0;0.3;0.6;0.9,2,5e-3;1e-3,20,5" \
#                                        -n fine_tuning \
# Usage: python hyperparameter_search.py -hp "sample_num,sample_step,lr_epoch" \
#                                        -v "5;10;15,1;2,5;10;15" \
#                                        -n RL_learning \
# Usage: python hyperparameter_search.py -hp "noise_var,ppo_clip,step_length" \
#                                        -v "0.02;0.04;0.06,0.15;0.25;0.35,0.5;0.75;1" \
#                                        -n RL_searching \


MODEL = "vgg16"
DATASET = "cifar100"

SPARSITY = 0.60

LOG = "log"
CKPT = "checkpoint"
PRETRAINED_MODEL_DIR = "pretrained_model"
COMPRESSED_MODEL_DIR = "compressed_model"
CKPT_DIR = f"{CKPT}/{MODEL}_{DATASET}"
PRETRAINED_MODEL_PTH = f"{PRETRAINED_MODEL_DIR}/{MODEL}_{DATASET}_original.pth"
COMPRESSED_MODEL_PTH = f"{COMPRESSED_MODEL_DIR}/{MODEL}_{DATASET}_{SPARSITY}.pth"

SUPPORT_HYPERPARAMETER = ["noise_var", "ppo_clip", "step_length",
                          "sample_num", "sample_step", "lr_epoch",
                          "greedy_epsilon", "prune_strategy",
                          "stu_co", "KD_temperature", "lr", "post_training_epoch", "warmup_epoch"]

def main():
    global args
    global epoch
    args = get_args()
    hyparameter_list = args.hyperparameter
    hyparameter_values = args.values
    PROJECT_NAME = args.project_name

    hyparameter_list_array = hyparameter_list.split(',')
    hyparameter_values_array = hyparameter_values.split(',')

    for parameter in hyparameter_list_array:
        if parameter not in SUPPORT_HYPERPARAMETER:
            print(f"Unsupported hyperparameter {parameter}")
            sys.exit(1)

    hyparameter_values_map = {hyparameter_list_array[i]: hyparameter_values_array[i]
                            for i in range(len(hyparameter_list_array))}

    hyperparameter_combinations = list(generate_combinations(hyparameter_values_map))

    """ Run all hyperparameter combinations """
    for combination in hyperparameter_combinations:
        cmd = [
            "python", "-m", "compress",
            "--model", MODEL,
            "--dataset", DATASET,
            "--device", "cuda",
            "--project_name", PROJECT_NAME,
            "--sparsity", str(SPARSITY),
            "--ppo",
            "--pretrained_dir", PRETRAINED_MODEL_DIR,
            "--compressed_dir", COMPRESSED_MODEL_DIR,
            "--checkpoint_dir", CKPT_DIR,
            "--log_dir", LOG,
            "--use_wandb",
            "--random_seed", "1"
        ]
        
        for param, value in combination.items():
            cmd.extend([f"--{param}", value])
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)


def generate_combinations(hyparameter_values_map):
        """ Generate all combinations of hyperparameter values """
        keys = hyparameter_values_map.keys()
        values = [v.split(';') for v in hyparameter_values_map.values()]
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))


def get_args():
    parser = argparse.ArgumentParser(description='train given model under given dataset')
    parser.add_argument('--hyperparameter', '-hp', type=str, default=None, 
                        help='the hyperparameter to search')
    parser.add_argument('--values', '-v', type=str, default=None, 
                        help='the values of hyperparameter to search')
    parser.add_argument('--project_name', '-n', type=str, default="RLPruner",
                        help='the independent project name of wandb of this searching for easy comparision')

    args = parser.parse_args()

    return args
    

if __name__ == '__main__':
    main()
