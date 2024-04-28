import os
import time
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import copy
from thop import profile

from conf import settings
from models.lenet import LeNet5
from models.vgg import VGG16
from models.googlenet import GoogleNet
from models.resnet import ResNet50
from models.unet import UNet
from utils import Custom_Conv2d, Custom_Linear, count_custom_conv2d, count_custom_linear, get_net_class, get_dataloader, setup_logging, WarmUpLR

def prune_architecture(top1_acc: float):
    global net
    global tolerance_times
    global modification_num
    global optimizer
    global accuracy_threshold
    global strategy
    if tolerance_times >= 0:
        net, model_index = copy.deepcopy(generate_architecture(net, top1_acc))
        net = net.to(device)
        if model_index == 0:
            tolerance_times -= 1
            if eac == True:
                if tolerance_times in settings.TOLERANCE_MILESTONES:
                    modification_num /= 2
            else:
                modification_num /= 2
        optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
        # save the module
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(net, f'models/{args.net}_{args.dataset}_Pruned_{start_time}.pkl')
    else:
        if args.criteria == "compression" and Para_compression_ratio >= compression_threshold:
            # decrement accuracy_threshold and reinitialize hyperparameter if criteria is compression ratio
            accuracy_threshold -= 0.05
            modification_num = settings.MAX_PRUNE_NUM
            tolerance_times = settings.MAX_TOLERANCE_TIMES
            logging.info('Current accuracy threshold: {}'.format(accuracy_threshold))
        else:
            strategy = "finished"
            modification_num = settings.MAX_QUANTIZE_NUM
            tolerance_times = settings.MAX_TOLERANCE_TIMES
            # save the module and break
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, f'models/{args.net}_{args.dataset}_Pruned_{start_time}.pkl')

def quantize_architecture(top1_acc: float):
    global net
    global tolerance_times
    global optimizer
    global accuracy_threshold
    global strategy
    if tolerance_times >= 0:
        net, model_index = copy.deepcopy(generate_architecture(net, top1_acc))
        net = net.to(device)
        if model_index == 0:
            tolerance_times -= 1
        optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
        # save the module
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(net, f'models/{args.net}_{args.dataset}_Quantized_{start_time}.pkl')
    else:
        strategy = "finished"
        # save the module and break
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(net, f'models/{args.net}_{args.dataset}_Quantized_{start_time}.pkl')


def train(epoch: int):
    net.train()
    with tqdm(total=len(training_loader), desc=f'Epoch {epoch}/{settings.DYNAMIC_EPOCH}', unit='batch') as pbar:
        for _, (images, labels) in enumerate(training_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

@torch.no_grad()
def eval_training(epoch: int):
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    for (images, labels) in tqdm(test_loader, total=len(test_loader), desc='Testing round', unit='batch', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(test_loader.dataset)
    top5_acc = correct_5 / len(test_loader.dataset)
    logging.info(f'Epoch: {epoch}, Top1 Accuracy: {top1_acc}, Top5 Accuracy: {top5_acc}')
    return top1_acc, top5_acc


def generate_architecture(original_net: nn.Module, local_top1_accuracy: float):
    global new_net_top1_accuracy_cache
    global prune_distribution_cache
    # initialize all evaluating variables
    top1_accuracy_tensors = torch.zeros(2)
    FLOPs_tensors = torch.zeros(2)
    parameter_num_tensors = torch.zeros(2)
    
    top1_accuracy_tensors[0] = local_top1_accuracy
    local_FLOPs, local_parameter_num = profile(original_net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    FLOPs_tensors[0] = local_FLOPs
    parameter_num_tensors[0] = local_parameter_num

    # generate architecture
    best_new_net = get_best_generated_architecture(original_net)
    if best_new_net == None:
        if eac == True:
            original_net.update_prune_distribution(new_net_top1_accuracy_cache, prune_distribution_cache, 
                                                        step_length, settings.PROBABILITY_LOWER_BOUND, settings.PPO_CLIP)
        logging.info('All generated architectures are worse than the architecture in caches. Stop fully training on them')
        logging.info(f'Current prune probability distribution: {original_net.prune_distribution}')
        return original_net, 0
    
    best_new_net = copy.deepcopy(best_new_net).to(device)
    dev_lr = lr
    dev_optimizer = optim.SGD(best_new_net.parameters(), lr=dev_lr, momentum=0.9, weight_decay=5e-4)
    dev_warmup_scheduler = WarmUpLR(dev_optimizer, iter_per_epoch * warm)

    for dev_epoch in range(1, dev_num + 1):
        if dev_epoch in settings.DYNAMIC_MILESTONES:
            dev_lr *= gamma
            for param_group in dev_optimizer.param_groups:
                param_group['lr'] = dev_lr
        
        best_new_net.train()   
        with tqdm(total=len(training_loader), desc=f'Training best generated Architecture Epoch {dev_epoch}/{dev_num}', unit='batch', leave=False) as pbar:
            for _, (images, labels) in enumerate(training_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                dev_optimizer.zero_grad()
                outputs = best_new_net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                dev_optimizer.step()

                if dev_epoch <= warm:
                    dev_warmup_scheduler.step()
                
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item()})


    correct_1 = 0.0
    best_new_net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = best_new_net(images)
            _, preds = outputs.topk(5, 1, largest=True, sorted=True)
            correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()

    top1_accuracy_tensors[1] = correct_1 / len(test_loader.dataset)
    dev_FLOPs, dev_parameter_num = profile(best_new_net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    FLOPs_tensors[1] = dev_FLOPs
    parameter_num_tensors[1] = dev_parameter_num
    
    # compare best_new_net with original net
    global Para_compression_ratio
    score_tensors = compute_score(top1_accuracy_tensors, FLOPs_tensors, parameter_num_tensors)
    best_net_index = torch.argmax(score_tensors)
    best_net_FLOPs = FLOPs_tensors[best_net_index]
    best_net_Params = parameter_num_tensors[best_net_index]
    FLOPs_compression_ratio = 1 - best_net_FLOPs / original_FLOPs_num
    Para_compression_ratio = 1 - best_net_Params / original_para_num

    if best_net_index == 0:
        optimal_net = original_net
        logging.info('Original net wins')
    else:
        optimal_net = best_new_net
        logging.info('Generated net wins')

    if eac == True:
        optimal_net.update_prune_distribution(new_net_top1_accuracy_cache, prune_distribution_cache, 
                                                    step_length, settings.PROBABILITY_LOWER_BOUND, settings.PPO_CLIP)

    if best_net_index == 1:
        # clear the cache when architecture has been updated
        new_net_top1_accuracy_cache = torch.zeros(generate_num)
        prune_distribution_cache = torch.zeros(generate_num, net.prune_choices_num)
    logging.info(f'Current compression ratio: FLOPs: {FLOPs_compression_ratio}, Parameter number {Para_compression_ratio}')
    logging.info(f'Current prune probability distribution: {optimal_net.prune_distribution}')
    
    return optimal_net, best_net_index


def get_best_generated_architecture(original_net):
    # initialize all evaluating variables
    net_list = []
    dev_top1_accuracy_tensors = torch.zeros(generate_num)

    with tqdm(total=generate_num, desc=f'Generated architectures', unit='model', leave=False) as pbar:
        for model_id in range(generate_num):
            # generate architecture
            generated_net = copy.deepcopy(original_net).to(device)
            dev_prune_distribution_tensor = net_class.update_architecture(generated_net, modification_num, strategy, 
                                                                                settings.NOISE_VAR, settings.PROBABILITY_LOWER_BOUND)
    
            correct_1 = 0.0
            generated_net.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = generated_net(images)
                    _, preds = outputs.topk(5, 1, largest=True, sorted=True)
                    correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()

            dev_top1_accuracy_tensors[model_id] = correct_1 / len(test_loader.dataset)

            pbar.update(1)
            net_list.append(generated_net)
            # only update cache when acc is higher in cache
            global new_net_top1_accuracy_cache
            global prune_distribution_cache
            min_acc, min_idx = torch.min(new_net_top1_accuracy_cache, dim=0)
            if dev_top1_accuracy_tensors[model_id] >= min_acc:
                new_net_top1_accuracy_cache[min_idx] = dev_top1_accuracy_tensors[model_id]
                prune_distribution_cache[min_idx] = dev_prune_distribution_tensor
    best_net_index = torch.argmax(dev_top1_accuracy_tensors)
    best_generated_net = net_list[best_net_index]

    logging.info('Generated net Top1 Accuracy List: {}'.format(dev_top1_accuracy_tensors))
    logging.info('Current new net top1 accuracy cache: {}'.format(new_net_top1_accuracy_cache))
    logging.info('Current prune probability distribution cache: {}'.format(prune_distribution_cache))
    logging.info(f'Net {best_net_index} is the best new net')
    if torch.max(dev_top1_accuracy_tensors) >= torch.max(new_net_top1_accuracy_cache) - 0.005:
        return best_generated_net
    else:
        return None


def compute_score(top1_accuracy_tensors, FLOPs_tensors, parameter_num_tensors):
    score_tensors = torch.zeros(2)
    
    if top1_accuracy_tensors[1] > accuracy_threshold - 0.005:
        score_tensors[1] = 1.0
    else:
        score_tensors = top1_accuracy_tensors
    
    logging.info(f'Generated Model Top1 Accuracy List: {top1_accuracy_tensors}')
    logging.info(f'Generated Model Parameter Number List: {parameter_num_tensors}')
    logging.info(f'Generated Model Score List: {score_tensors}')
    return score_tensors


def get_args():
    parser = argparse.ArgumentParser(description='Adaptively compress the given trained model under given dataset')
    parser.add_argument('-net', type=str, default=None, help='the type of model to train')
    parser.add_argument('-dataset', type=str, default=None, help='the dataset to train on')
    parser.add_argument('-warm', type=int, default=settings.WARM, help='warm up training phase')
    parser.add_argument('-crit', '--criteria', type=str, default='accuracy', help='compressed the model with accuracy_threshold or compression_threshold')
    parser.add_argument('-at', '--accuracy_threshold', type=float, default=None, help='the final accuracy the architecture will achieve')
    parser.add_argument('-ct', '--compression_threshold', type=float, default=None, help='the final compression ratio the architecture will achieve')
    parser.add_argument('-eac', '--enable_adptive_compression', action='store_true', default=False, help='Enable adaptive compress if set')
    parser.add_argument('-lr', type=float, default=settings.INITIAL_LR, help='initial learning rate')
    parser.add_argument('-lr_decay', type=float, default=settings.LR_DECAY, help='learning rate decay rate')
    parser.add_argument('-b', type=int, default=settings.BATCH_SIZE, help='batch size for dataloader')

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args: argparse.Namespace):
    if args.net is None:
        logging.error("the specific type of model should be provided, please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
        sys.exit(1)
    elif args.net not in ['lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet']:
        logging.error('the specific model is not supported')
        sys.exit(1)
    if args.dataset is None:
        logging.error("the specific type of dataset to train on should be provided, please select one of 'mnist', 'cifar10', 'cifar100', 'imagenet'")
        sys.exit(1)
    elif args.dataset not in ['mnist', 'cifar10', 'cifar100', 'imagenet']:
        logging.error('the specific dataset is not supported')
        sys.exit(1)
    
    if args.criteria == 'accuracy':
        if args.compression_threshold is not None:
            logging.error("--compression_threshold is not allowed when criteria is 'accuracy'")
            sys.exit(1)
        if args.accuracy_threshold is None:
            logging.error("accuracy_threshold should be provided when criteria is 'accuracy'")
            sys.exit(1)
    elif args.criteria == 'compression':
        if args.accuracy_threshold is not None:
            logging.error("--accuracy_threshold is not allowed when criteria is 'compression'")
            sys.exit(1)
        if args.compression_threshold is None:
            logging.error("compression_threshold should be provided when criteria is 'compression'")
            sys.exit(1)
    else:
        logging.error("--criteria must be 'accuracy' or 'compression'")
        sys.exit(1)


if __name__ == '__main__':
    start_time = int(time.time())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    setup_logging(experiment_id=start_time, net=args.net, dataset=args.dataset, action='compress')
    
    # reinitialize random seed
    torch.manual_seed(start_time)
    logging.info('Start with random seed: {}'.format(start_time))

    # process input arguments
    training_loader, prototyping_loader, test_loader, _, _ = get_dataloader(dataset=args.dataset, batch_size=args.b)
    net_class = get_net_class(args.net)
    net = torch.load('models/vgg16_cifar100_Original_1714258157.pkl').to(device)  # replace it with the model gained by train.py
    lr = args.lr
    gamma = args.lr_decay
    warm = args.warm
    
    # set experiment parameter
    current_lr = lr * gamma * gamma * gamma
    best_acc = 0.0
    step_length = settings.STEP_LENGTH
    generate_num = settings.MAX_GENERATE_NUM
    modification_num = settings.MAX_PRUNE_NUM
    eac = args.enable_adptive_compression
    if eac == True:
        tolerance_times = settings.MAX_TOLERANCE_TIMES_EAC
    else:
        tolerance_times = settings.MAX_TOLERANCE_TIMES
    strategy = "prune"
    dev_num = settings.DEV_NUM
    dev_pretrain_num = settings.DEV_PRETRAIN_NUM
    if args.criteria == 'accuracy':
        accuracy_threshold = args.accuracy_threshold
        compression_threshold = settings.DEFAULT_COMPRESSION_THRESHOLD
    else:
        accuracy_threshold = settings.DEFAULT_ACCURACY_THRESHOLD
        compression_threshold = args.compression_threshold

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)


    # compute inital data
    if args.net == 'lenet5':
        input = torch.rand(1, 1, 32, 32).to(device)
    else:
        input = torch.rand(1, 3, 32, 32).to(device)
    custom_ops = {Custom_Conv2d: count_custom_conv2d, Custom_Linear: count_custom_linear}
    original_FLOPs_num, original_para_num = profile(net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    Para_compression_ratio = 1.0
    new_net_top1_accuracy_cache = torch.zeros(generate_num)
    prune_distribution_cache = torch.zeros(generate_num, net.prune_choices_num)
    logging.info(f'Initial prune probability distribution: {net.prune_distribution}')

    
    for epoch in range(1, settings.DYNAMIC_EPOCH + 1):
        if strategy == "finished":
            break
        
        train(epoch)
        top1_acc, _ = eval_training(epoch)

        # dynamic generate architecture
        if epoch % 5 == 0:
            if strategy == "prune":
                prune_architecture(top1_acc)
            elif strategy == "quantize":
                quantize_architecture(top1_acc)
            elif strategy == "finished":
                break
            else:
                raise TypeError("strategy must be 'prune' or 'quantize' or 'finished' ")

        # start to save best performance model after first pruning milestone
        if tolerance_times < settings.TOLERANCE_MILESTONES[1] and best_acc < top1_acc:
            best_acc = top1_acc

            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, f'models/{args.net}_{args.dataset}_Compressed_{start_time}.pkl')

        if epoch == settings.DYNAMIC_EPOCH:
            # save the module
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, f'models/{args.net}_{args.dataset}_Compressed_{start_time}.pkl')
    
    end_time = int(time.time())
    logging.info(f'Original training process takes {(end_time - start_time) / 60} minutes')