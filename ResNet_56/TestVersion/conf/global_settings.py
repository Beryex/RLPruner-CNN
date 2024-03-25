#mean and std of cifar100 dataset
CIFAR10_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR10_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# global hyperparameter
# for compression algorithm
MAX_GENERATE_NUM = 2        # for each updates, how many potential architecture we are going to generate
MAX_TOLERANCE_TIMES = 3     # for each training, how many updates we are going to apply before we get the final architecture
MAX_MODIFICATION_NUM = 100  # max update numbers, that is max modification we make to architecture in update_architecture
DEV_NUM = 20                # for each potential architecture, how many epochs we are going to train it
DEFAULT_ACCURACY_THRESHOLD = 0.705  # if current top1 accuracy is above the accuracy_threshold, then computation of architecture's score main focus on FLOPs and parameter #
DEFAULT_COMPRESSION_THRESHOLD = 1   # if current top1 accuracy is above the accuracy_threshold, then computation of architecture's score main focus on FLOPs and parameter #


# for training parameters
ORIGINAL_EPOCH = 130
DYNAMIC_EPOCH = 400
ORIGINAL_MILESTONES = [40, 80, 100]
DYNAMIC_MILESTONES = [8, 15, 19]
