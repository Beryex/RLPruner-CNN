# hyperparameter for Reinforcement Learning
RL_STEP_LENGTH = 0.08
RL_PRUNE_FILTER_NOISE_VAR = 0.05

RL_PPO_CLIP = 0.2
RL_PPO_ENABLE = True
RL_PROBABILITY_LOWER_BOUND = 1e-5

RL_GREEDY_EPSILON = 0

RL_MAX_SAMPLE_STEP = 2
RL_MAX_GENERATE_NUM = 8
RL_GENERATE_NUM_SCALING_FACTOR = 2
RL_DISCOUNT_FACTOR = 0.9
RL_LR_EPOCH = 15
RL_LR_TOLERANCE = 3
RL_REWARD_CHANGE_THRESHOLD = 0.003
RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT = 0.8


# hyperparameter for dataset
D_MNIST_TRAIN_MEAN = (0.1307, )
D_MNIST_TRAIN_STD = (0.3081, )

D_CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215827, 0.44653124)
D_CIFAR10_TRAIN_STD = (0.24703233, 0.24348505, 0.26158768)

D_CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
D_CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

D_VAL_PROPORTION = 0


# hyperparameter for training
T_LR_SCHEDULAR_MIN_LR = 1e-6
T_LR_SCHEDULAR_INITIAL_LR = 1e-1
T_FT_LR_SCHEDULAR_INITIAL_LR = 1e-3

T_BATCH_SIZE = 256
T_NUM_WORKERS = 8
T_WARM = 1
T_ORIGINAL_EPOCH = 250


# hyperparameter for compressing
C_COS_PRUNE_EPOCH = 45
C_COMPRESSION_EPOCH = 50
C_COS_DEV_NUM = 20
C_DEV_NUM = 25
C_FT_EPOCH = 100

C_SINGLE_STEP_ACCURACY_CHANGE_THRESHOLD = 0.003
C_SINGLE_STEP_ACCURACY_CHANGE_THRESHOLD_INCRE = 0.001
C_OVERALL_ACCURACY_CHANGE_THRESHOLD = 0.01

C_PRUNE_FILTER_MAX_RATIO = 0.08
C_PRUNE_FILTER_MIN_RATIO = 0.01             
