import os
import time
import argparse
import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import copy
from thop import profile
import torch.multiprocessing as mp

from conf import settings
from utils import (Custom_Conv2d, Custom_Linear, count_custom_conv2d, count_custom_linear, get_net_class, 
                   get_dataloader, setup_logging, WarmUpLR, Prune_agent, torch_set_random_seed)

def prune_architecture(net: nn.Module, 
                       top1_acc: float, 
                       prune_agent: Prune_agent):
    
    net, model_index = copy.deepcopy(get_optimal_architecture(net, top1_acc, prune_agent))
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    ret = prune_agent.step(model_index)
    if model_index == 1: # means generated architecture is better
        # save the module
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(net, f'models/{args.net}_{args.dataset}_{prune_agent.strategy}_{start_time}.pkl')
    
    if ret is None:
        if prune_agent.strategy == "prune_filter":
            prune_agent.change_strategy("weight_sharing", net.prune_choices_num)
        elif prune_agent.strategy == "weight_sharing":
            # free all initial weights
            for idx, layer_idx in enumerate(net.prune_choices):
                if idx <= net.last_conv_layer_idx:
                    layer = net.conv_layers[layer_idx]
                else:
                    layer = net.linear_layers[layer_idx]
                layer.free_original_weight
            prune_agent.change_strategy("finished", -1)
    return net, optimizer


def train_network(target_net: nn.Module, 
          target_optimizer: optim.Optimizer, 
          target_training_loader: torch.utils.data.DataLoader, 
          loss_function: nn.Module, 
          warmup_scheduler: WarmUpLR,
          epoch: int, 
          dev: bool = False,
          comment: str = ''):
    
    target_net.train()
    if dev == False:
        Max_epoch = settings.DYNAMIC_EPOCH
        leave = True
    else:
        Max_epoch = settings.DEV_NUM
        leave = False
        comment = 'Training best generated Architecture Epoch'
    
    with tqdm(total=len(target_training_loader), desc=f'{comment} Epoch {epoch}/{Max_epoch}', unit='batch', leave=leave) as pbar:
        for _, (images, labels) in enumerate(target_training_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            target_optimizer.zero_grad()
            outputs = target_net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            target_optimizer.step()

            if dev == True and epoch <= warm:
                warmup_scheduler.step()

            pbar.update(1)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

@torch.no_grad()
def eval_network(target_net: nn.Module, 
                  target_test_loader: torch.utils.data.DataLoader,
                  epoch: int,
                  dev: bool = False):
    
    target_net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    for (images, labels) in tqdm(target_test_loader, total=len(target_test_loader), desc='Testing round', unit='batch', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = target_net(images)
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(target_test_loader.dataset)
    top5_acc = correct_5 / len(target_test_loader.dataset)
    if dev == False:
        logging.info(f'Epoch: {epoch}, Top1 Accuracy: {top1_acc}, Top5 Accuracy: {top5_acc}')
    return top1_acc, top5_acc


def get_optimal_architecture(original_net: nn.Module, 
                             local_top1_accuracy: float, 
                             prune_agent: Prune_agent):
    # initialize all evaluating variables
    top1_accuracy_tensors = torch.zeros(2)
    FLOPs_tensors = torch.zeros(2)
    parameter_num_tensors = torch.zeros(2)
    
    top1_accuracy_tensors[0] = local_top1_accuracy
    local_FLOPs, local_parameter_num = profile(original_net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    FLOPs_tensors[0] = local_FLOPs
    parameter_num_tensors[0] = local_parameter_num

    # generate architecture
    best_new_net = get_best_generated_architecture(original_net, prune_agent)
    if best_new_net == None:
        prune_agent.update_prune_distribution(settings.STEP_LENGTH, settings.PROBABILITY_LOWER_BOUND, settings.PPO_CLIP)
        logging.info('All generated architectures are worse than the architecture in caches. Stop fully training on them')
        logging.info(f'Current prune probability distribution: {prune_agent.prune_distribution}')
        return original_net, 0
    
    if prune_agent.strategy == "prune_filter":
        dev_lr = lr
        dev_optimizer = optim.SGD(best_new_net.parameters(), lr=dev_lr, momentum=0.9, weight_decay=5e-4)
        dev_warmup_scheduler = WarmUpLR(dev_optimizer, iter_per_epoch * warm)

        for dev_epoch in range(1, dev_num + 1):
            if dev_epoch in settings.DYNAMIC_MILESTONES:
                dev_lr *= lr_decay
                for param_group in dev_optimizer.param_groups:
                    param_group['lr'] = dev_lr
            
            train_network(best_new_net, dev_optimizer, training_loader, loss_function, dev_warmup_scheduler, dev_epoch, dev=True, comment='Training best generated Architecture Epoch')

    top1_acc,_ = eval_network(best_new_net, test_loader, epoch=-1, dev=True)

    top1_accuracy_tensors[1] = top1_acc
    dev_FLOPs, dev_parameter_num = profile(best_new_net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    FLOPs_tensors[1] = dev_FLOPs
    parameter_num_tensors[1] = dev_parameter_num
    
    # compare best_new_net with original net
    global Para_compression_ratio
    score_tensors = compute_score(top1_accuracy_tensors)
    optimal_net_index = torch.argmax(score_tensors)

    # Use Epsilon-greedy exploration strategy
    if torch.rand(1).item() < settings.GREEDY_EPSILON:
        if torch.rand(1).item() < 0.5:
            optimal_net = original_net
            optimal_net_index = 0
            logging.info('Exploration: Original net wins')
        else:
            optimal_net = best_new_net
            optimal_net_index = 1
            logging.info('Exploration: Generated net wins')
    else:
        if optimal_net_index == 0:
            optimal_net = original_net
            logging.info('Exploitation: Original net wins')
        else:
            optimal_net = best_new_net
            logging.info('Exploitation: Generated net wins')
    
    optimal_net_FLOPs = FLOPs_tensors[optimal_net_index]
    optimal_net_Params = parameter_num_tensors[optimal_net_index]
    FLOPs_compression_ratio = 1 - optimal_net_FLOPs / original_FLOPs_num
    Para_compression_ratio = 1 - optimal_net_Params / original_para_num

    prune_agent.update_prune_distribution(settings.STEP_LENGTH, settings.PROBABILITY_LOWER_BOUND, settings.PPO_CLIP)

    logging.info(f'Current compression ratio: FLOPs: {FLOPs_compression_ratio}, Parameter number {Para_compression_ratio}')
    logging.info(f'Current prune probability distribution: {prune_agent.prune_distribution}')
    
    return optimal_net, optimal_net_index


def get_best_generated_architecture(original_net: nn.Module,
                                    prune_agent: Prune_agent):
    # initialize all evaluating variables
    net_list = []

    with tqdm(total=generate_num, desc=f'Generated architectures', unit='model', leave=False) as pbar:
        # Q_value directly equal to Rewards as we initial all Q_value to 0 and only forward 1 time step
        Q_value_1 = torch.zeros(generate_num)
        for model_id_1 in range(generate_num):
            # generate architecture level 1
            generated_net_1 = copy.deepcopy(original_net).to(device)
            prune_distribution_action_1, prune_counter_1 = net_class.update_architecture(generated_net_1, prune_agent, settings.PROBABILITY_LOWER_BOUND)
            top1_acc_1,_ = eval_network(generated_net_1, test_loader, epoch=-1, dev=True)

            Q_value_1[model_id_1] = top1_acc_1

            Q_value_2 = torch.zeros(generate_num)
            for model_id_2 in range(generate_num):
                # generate architecture level 2
                generated_net_2 = copy.deepcopy(generated_net_1).to(device)
                prune_distribution_action_2, prune_counter_2 = net_class.update_architecture(generated_net_2, prune_agent, settings.PROBABILITY_LOWER_BOUND)
                # use Reward cache to avoid repetition
                key = prune_counter_1.int() + prune_counter_2.int()
                tuple_key = tuple(key.tolist())
                if tuple_key in prune_agent.Reward_cache:
                    Q_value_2[model_id_2] = prune_agent.Reward_cache[tuple_key]
                    continue
                
                top1_acc_2,_ = eval_network(generated_net_2, test_loader, epoch=-1, dev=True)
                Q_value_2[model_id_2] = top1_acc_2
                
                prune_agent.Reward_cache[tuple_key] = Q_value_2[model_id_2]

            Q_value_1[model_id_1] += settings.DISCOUNT_FACTOR * torch.max(Q_value_2)

            pbar.update(1)
            net_list.append(generated_net_1)

            # only update cache when acc is higher in cache
            min_acc, min_idx = torch.min(prune_agent.ReplayBuffer[:, 0], dim=0)
            if Q_value_1[model_id_1] >= min_acc:
                prune_agent.ReplayBuffer[min_idx, 0] = Q_value_1[model_id_1]
                prune_agent.ReplayBuffer[min_idx, 1:] = prune_distribution_action_1
    best_net_index = torch.argmax(Q_value_1)
    best_generated_net = net_list[best_net_index]

    logging.info('Generated net Q value List: {}'.format(Q_value_1))
    logging.info('Current new net Q value cache: {}'.format(prune_agent.ReplayBuffer[:, 0]))
    logging.info('Current prune probability distribution cache: {}'.format(prune_agent.ReplayBuffer[:, 1:]))
    logging.info(f'Net {best_net_index} is the best new net')
    return best_generated_net


def compute_score(top1_accuracy_tensors: Tensor):
    score_tensors = torch.zeros(2)
    
    if top1_accuracy_tensors[1] > accuracy_threshold - 0.005:
        score_tensors[1] = 1.0
    else:
        score_tensors = top1_accuracy_tensors
    
    logging.info(f'Generated Model Top1 Accuracy List: {top1_accuracy_tensors}')
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
    parser.add_argument('-lr', type=float, default=settings.INITIAL_LR, help='initial learning rate')
    parser.add_argument('-lr_decay', type=float, default=settings.LR_DECAY, help='learning rate decay rate')
    parser.add_argument('-b', type=int, default=settings.BATCH_SIZE, help='batch size for dataloader')
    parser.add_argument('-n', type=int, default=settings.NUM_WORKERS, help='num_workers for dataloader')

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
    start_time = 6
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    setup_logging(experiment_id=start_time, net=args.net, dataset=args.dataset, action='compress')
    
    # reinitialize random seed
    torch_set_random_seed(start_time)
    logging.info('Start with random seed: {}'.format(start_time))

    # process input arguments
    training_loader, prototyping_loader, test_loader, _, _ = get_dataloader(dataset=args.dataset, batch_size=args.b, num_workers=args.n, pin_memory=True)
    net_class = get_net_class(args.net)
    net = torch.load('models/vgg16_cifar100_Original_1716689096.pkl').to(device)  # replace it with the model gained by train.py
    lr = args.lr
    lr_decay = args.lr_decay
    warm = args.warm
    
    # initialize lr rate
    current_lr = lr * lr_decay * lr_decay * lr_decay
    best_acc = 0.0

    # initialize compress parameter
    generate_num = settings.MAX_GENERATE_NUM
    dev_num = settings.DEV_NUM
    dev_pretrain_num = settings.DEV_PRETRAIN_NUM

    # initialize compress milestone
    tolerance_times = settings.MAX_TOLERANCE_TIMES
    if args.criteria == 'accuracy':
        accuracy_threshold = args.accuracy_threshold
        compression_threshold = settings.DEFAULT_COMPRESSION_THRESHOLD
    else:
        accuracy_threshold = settings.DEFAULT_ACCURACY_THRESHOLD
        compression_threshold = args.compression_threshold

    # compute inital data
    if args.net == 'lenet5':
        input = torch.rand(1, 1, 32, 32).to(device)
    else:
        input = torch.rand(1, 3, 32, 32).to(device)
    custom_ops = {Custom_Conv2d: count_custom_conv2d, Custom_Linear: count_custom_linear}
    original_FLOPs_num, original_para_num = profile(net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    Para_compression_ratio = 1.0

    # initialize training parameter
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    # initialize reinforcement learning parameter
    prune_distribution = torch.zeros(net.prune_choices_num)
    for idx, layer_idx in enumerate(net.prune_choices):
        if idx <= net.last_conv_layer_idx:
            layer = net.conv_layers[layer_idx]
            prune_distribution[idx] = layer.out_channels
        else:
            layer = net.linear_layers[layer_idx]
            prune_distribution[idx] = layer.out_features
    filter_num = torch.sum(prune_distribution)
    prune_distribution = prune_distribution / filter_num
    ReplayBuffer = torch.zeros(generate_num, 1 + net.prune_choices_num) # [:, 0] stores Q(s, a), [:, 1:] stores action a
    prune_agent = Prune_agent(strategy="prune_filter", prune_distribution=prune_distribution, 
                                        ReplayBuffer=ReplayBuffer, filter_num=filter_num, prune_choices_num=net.prune_choices_num)
    logging.info(f'Initial prune probability distribution: {prune_agent.prune_distribution}')

    
    for epoch in range(1, settings.DYNAMIC_EPOCH + 1):
        if prune_agent.strategy == "finished":
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, f'models/{args.net}_{args.dataset}_Compressed_{start_time}.pkl')
            break
        elif prune_agent.strategy == "prune_filter":
            train_network(net, optimizer, training_loader, loss_function, warmup_scheduler, epoch=epoch, dev=False, comment='')
            top1_acc, _ = eval_network(net, test_loader, epoch=epoch, dev=False)

            if best_acc < top1_acc:
                best_acc = top1_acc

                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(net, f'models/{args.net}_{args.dataset}_{prune_agent.strategy}_{start_time}.pkl')

            if epoch % 5 == 0:
                net, optimizer = prune_architecture(net, top1_acc, prune_agent)
        elif prune_agent.strategy == "weight_sharing":
            top1_acc, _ = eval_network(net, test_loader, epoch=epoch, dev=False)
            net, optimizer = prune_architecture(net, top1_acc, prune_agent)
        
    
    end_time = int(time.time())
    logging.info(f'Compress process takes {(end_time - start_time) / 60} minutes')
