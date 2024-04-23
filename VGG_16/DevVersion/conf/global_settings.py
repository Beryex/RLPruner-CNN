#mean and std of cifar100 dataset
CIFAR10_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR10_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# global hyperparameter
# for compression algorithm
STEP_LENGTH = 0.2
MAX_GENERATE_NUM = 4        # for each updates, how many potential architecture we are going to generate
MAX_TOLERANCE_TIMES = 3     # for each training, how many updates we are going to apply before we get the final architecture
MAX_TOLERANCE_TIMES_EAP = 10
MAX_PRUNE_NUM = 400         # max pruning numbers, that is max pruning we make to architecture in update_architecture
MAX_QUANTIZE_NUM = 20       # max quantize numbers, that is max quantize we make to architecture in update_architecture
DEV_PRETRAIN_NUM = 10       # for each potential architecture, how many epochs we are going to train it
DEV_NUM = 20                # for each potential architecture, how many epochs we are going to train it
DEFAULT_ACCURACY_THRESHOLD = 0.715  # if current top1 accuracy is above the accuracy_threshold, then computation of architecture's score main focus on FLOPs and parameter #
DEFAULT_COMPRESSION_THRESHOLD = 1   # if current top1 accuracy is above the accuracy_threshold, then computation of architecture's score main focus on FLOPs and parameter #
DATASET_PROPORTION = 0.2


# for training parameters
INITIAL_LR = 0.1
LR_DECAY = 0.2
BATCH_SIZE = 256
ORIGINAL_EPOCH = 130
DYNAMIC_EPOCH = 200
ORIGINAL_MILESTONES = [40, 80, 100]
DYNAMIC_PRETRAIN_MILESTONES = [4, 7, 9]
DYNAMIC_MILESTONES = [8, 15, 19]
TOLERANCE_MILESTONES = [3, 6, 9]
