# global hyperparameter
# for compression algorithm
MAX_GENERATE_NUM = 2        # for each updates, how many potential architecture we are going to generate
MAX_TOLERANCE_TIMES = 3     # for each training, how many updates we are going to apply before we get the final architecture
MAX_MODIFICATION_NUM = 40   # max update numbers, that is max modification we make to architecture in update_architecture
DEV_NUM = 16                # for each potential architecture, how many epochs we are going to train it
DEFAULT_ACCURACY_THRESHOLD = 0.99  # if current top1 accuracy is above the accuracy_threshold, then computation of architecture's score main focus on FLOPs and parameter #
DEFAULT_COMPRESSION_THRESHOLD = 1   # if current top1 accuracy is above the accuracy_threshold, then computation of architecture's score main focus on FLOPs and parameter #


# for training parameters
ORIGINAL_EPOCH = 20
DYNAMIC_EPOCH = 300
ORIGINAL_MILESTONES = [10, 16]
DYNAMIC_MILESTONES = [8, 12]
