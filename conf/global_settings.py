# hyperparameter for Reinforcement Learning
RL_STEP_LENGTH = 0.08
RL_PRUNE_FILTER_NOISE_VAR = 0.025

RL_PPO_CLIP = 0.2
RL_PPO_ENABLE = True
RL_PROBABILITY_LOWER_BOUND = 1e-5

RL_GREEDY_EPSILON = 0

RL_MAX_SAMPLE_STEP = 1
RL_MAX_SAMPLE_NUM = 8
RL_DISCOUNT_FACTOR = 0.9
RL_LR_EPOCH = 10


# hyperparameter for dataset
D_MNIST_TRAIN_MEAN = (0.1307, )
D_MNIST_TRAIN_STD = (0.3081, )

D_CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215827, 0.44653124)
D_CIFAR10_TRAIN_STD = (0.24703233, 0.24348505, 0.26158768)

D_CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
D_CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

D_VAL_PROPORTION = 0


# hyperparameter for training
T_EPOCH = 250
T_LR_SCHEDULER_INITIAL_LR = 1e-1
T_LR_SCHEDULER_MIN_LR = 1e-6

T_FT_EPOCH = 20
T_FT_LR_SCHEDULER_INITIAL_LR = 5e-3
T_FT_STU_CO = 0

T_BATCH_SIZE = 256
T_NUM_WORKER = 8
T_WARMUP_EPOCH = 5


# hyperparameter for compressing
C_PRUNE_STRATEGY = "variance"
C_PRUNE_FILTER_RATIO = 0.01
C_COMPRESSION_EPOCH = 50         
