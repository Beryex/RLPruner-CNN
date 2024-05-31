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
                   get_dataloader, setup_logging, Prune_agent, torch_set_random_seed)


def free_initial_weight(net: nn.Module):
    # free all initial weights
    for idx, layer_idx in enumerate(net.prune_choices):
        if idx <= net.last_conv_layer_idx:
            layer = net.conv_layers[layer_idx]
        else:
            layer = net.linear_layers[layer_idx]
        layer.free_original_weight()


def fine_tuning_network(target_net: nn.Module, 
          target_optimizer: optim.Optimizer, 
          target_train_loader: torch.utils.data.DataLoader, 
          loss_function: nn.Module, 
          epoch: int):
    
    target_net.train()
    train_loss = 0.0
    with tqdm(total=len(target_train_loader), desc=f'Training best generated Architecture: Epoch {epoch}/{settings.DEV_NUM}', unit='batch', leave=False) as pbar:
        for _, (images, labels) in enumerate(target_train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            target_optimizer.zero_grad()
            outputs = target_net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            target_optimizer.step()
            train_loss += loss.item()

            pbar.update(1)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
        train_loss /= len(target_train_loader)
        pbar.set_postfix(**{'loss (batch)': train_loss})
    return train_loss

@torch.no_grad()
def eval_network(target_net: nn.Module, 
                 target_eval_loader: torch.utils.data.DataLoader,
                 loss_function: nn.Module,
                 epoch: int,
                 dev: bool = False):
    
    target_net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    eval_loss = 0.0
    for (images, labels) in tqdm(target_eval_loader, total=len(target_eval_loader), desc='Testing round', unit='batch', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = target_net(images)
        eval_loss += loss_function(outputs, labels).item()
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(target_eval_loader.dataset)
    top5_acc = correct_5 / len(target_eval_loader.dataset)
    eval_loss /= len(target_eval_loader)
    if dev == False:
        logging.info(f'Epoch: {epoch}/{settings.COMPRESSION_EPOCH}, Top1 Accuracy: {top1_acc}, Top5 Accuracy: {top5_acc}')
    return top1_acc, top5_acc, eval_loss


def prune_architecture(net: nn.Module, 
                       top1_acc: float, 
                       prune_agent: Prune_agent):
    
    net, model_index = get_optimal_architecture(net, top1_acc, prune_agent)
    ret = prune_agent.step(model_index)
    if model_index == 1: # means generated architecture is better
        # save the module
        torch.save(net, f'models/{args.net}_{args.dataset}_{prune_agent.strategy}_{start_time}.pth')
    
    if ret is None:
        if prune_agent.strategy == "prune_filter":
            prune_agent.change_strategy("weight_sharing", net.prune_choices_num)
        elif prune_agent.strategy == "weight_sharing":
            prune_agent.change_strategy("finished", -1)
    return net

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
    
    # fine tuning best new architecture
    if prune_agent.strategy == "prune_filter":
        optimizer = optim.SGD(best_new_net.parameters(), lr=settings.FT_LR_SCHEDULAR_INITIAL_LR, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=settings.LR_DECAY, threshold=settings.FT_LR_SCHEDULAR_THRESHOLD, 
                                                            patience=settings.FT_LR_SCHEDULAR_PATIENCE, threshold_mode='abs', min_lr=settings.LR_SCHEDULAR_MIN_LR, verbose=False)
        best_acc = 0.0
        for dev_epoch in range(1, dev_num + 1):
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            if current_lr <= settings.LR_SCHEDULAR_MIN_LR:
                break
            
            train_loss = fine_tuning_network(best_new_net, optimizer, train_loader, loss_function, dev_epoch)
            top1_acc, _, _ = eval_network(best_new_net, test_loader, loss_function, epoch=-1, dev=True)
            lr_scheduler.step(train_loss)

            if best_acc < top1_acc:
                best_acc = top1_acc
                torch.save(best_new_net, f'models/temporary_net.pth')
        best_new_net = torch.load('models/temporary_net.pth').to(device)

    top1_acc,_, _ = eval_network(best_new_net, test_loader, loss_function, epoch=-1, dev=True)

    top1_accuracy_tensors[1] = top1_acc
    dev_FLOPs, dev_parameter_num = profile(best_new_net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    FLOPs_tensors[1] = dev_FLOPs
    parameter_num_tensors[1] = dev_parameter_num
    
    # compare best_new_net with original net
    global FLOPs_compression_ratio
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
    if prune_agent.strategy == "prune_filter":
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
    Q_value_dict = {}

    sample_trajectory(cur_step=0,
                      original_net=original_net,
                      prune_agent=prune_agent,
                      Q_value_dict=Q_value_dict,
                      prev_prune_counter_sum=0,
                      net_list=net_list)
    
    best_net_index = torch.argmax(Q_value_dict[0])
    best_generated_net = net_list[best_net_index]

    logging.info('Generated net Q value List: {}'.format(Q_value_dict[0]))
    logging.info('Current new net Q value cache: {}'.format(prune_agent.ReplayBuffer[:, 0]))
    logging.info('Current prune probability distribution cache: {}'.format(prune_agent.ReplayBuffer[:, 1:]))
    logging.info(f'Net {best_net_index} is the best new net')
    return best_generated_net

def sample_trajectory(cur_step: int, 
                      original_net: nn.Module, 
                      prune_agent: Prune_agent, 
                      Q_value_dict: dict, 
                      prev_prune_counter_sum: int, 
                      net_list: list):
    # sample trajectory using DFS
    if cur_step == settings.MAX_SAMPLE_STEP:
        return

    cur_generate_num = max(generate_num // (settings.GENERATE_NUM_SCALING_FACTOR ** cur_step), 1)
    Q_value_dict[cur_step] = torch.zeros(cur_generate_num)

    with tqdm(total=cur_generate_num, desc=f'Generated architectures', unit='model', leave=False) as pbar:
        for model_id in range(cur_generate_num):
            # generate architecture
            generated_net = copy.deepcopy(original_net).to(device)
            prune_distribution_action, prune_counter = net_class.update_architecture(generated_net, prune_agent, settings.PROBABILITY_LOWER_BOUND)

            # evaluate generated architecture
            cur_prune_counter_sum = prune_counter.int() + prev_prune_counter_sum
            tuple_key = tuple(cur_prune_counter_sum.tolist())
            if tuple_key not in prune_agent.Reward_cache:
                top1_acc, _, eval_loss = eval_network(generated_net, test_loader, loss_function, epoch=-1, dev=True)
                Q_value_dict[cur_step][model_id] = top1_acc
                prune_agent.Reward_cache[tuple_key] = top1_acc
            else:
                Q_value_dict[cur_step][model_id] = prune_agent.Reward_cache[tuple_key]
            sample_trajectory(cur_step + 1, generated_net, prune_agent, Q_value_dict, cur_prune_counter_sum, net_list)

            if cur_step + 1 in Q_value_dict:
                Q_value_dict[cur_step][model_id] += settings.DISCOUNT_FACTOR * torch.max(Q_value_dict[cur_step + 1])
            
            # update Q_value and ReplayBuffer at top level
            if cur_step == 0:
                net_list.append(generated_net)
                
                new_trajectory = torch.cat((torch.tensor([Q_value_dict[0][model_id]]), prune_distribution_action), dim=0).unsqueeze(0)
                prune_agent.ReplayBuffer = torch.cat((prune_agent.ReplayBuffer, new_trajectory), dim=0)

            pbar.update(1)


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
    parser.add_argument('-crit', '--criteria', type=str, default='accuracy', help='compressed the model with accuracy_threshold or compression_threshold')
    parser.add_argument('-at', '--accuracy_threshold', type=float, default=None, help='the final accuracy the architecture will achieve')
    parser.add_argument('-ct', '--compression_threshold', type=float, default=None, help='the final compression ratio the architecture will achieve')
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
    start_time = 30
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    setup_logging(experiment_id=start_time, net=args.net, dataset=args.dataset, action='compress')
    
    # reinitialize random seed
    torch_set_random_seed(start_time)
    logging.info('Start with random seed: {}'.format(start_time))

    # process input arguments
    train_loader, prototyping_loader, test_loader, _, _ = get_dataloader(dataset=args.dataset, batch_size=args.b, num_workers=args.n, pin_memory=True)
    net_class = get_net_class(args.net)
    net = torch.load('models/vgg16_cifar100_Original_101.pth').to(device)  # replace it with the model gained by train.py
    

    # initialize compress parameter
    generate_num = settings.MAX_GENERATE_NUM
    dev_num = settings.DEV_NUM

    # initialize compress milestone
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
    FLOPs_compression_ratio = 1.0

    # initialize training parameter
    loss_function = nn.CrossEntropyLoss()

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
    ReplayBuffer = torch.tensor([]) # [:, 0] stores Q(s, a), [:, 1:] stores action a
    prune_agent = Prune_agent(strategy="prune_filter", prune_distribution=prune_distribution, 
                                        ReplayBuffer=ReplayBuffer, filter_num=filter_num, prune_choices_num=net.prune_choices_num)
    logging.info(f'Initial prune probability distribution: {prune_agent.prune_distribution}')

    
    for epoch in range(1, settings.COMPRESSION_EPOCH + 1):
        if prune_agent.strategy == "finished":
            free_initial_weight(net)
            torch.save(net, f'models/{args.net}_{args.dataset}_{prune_agent.strategy}_{start_time}.pth')
            break
        elif prune_agent.strategy == "prune_filter":
            top1_acc, _, _ = eval_network(net, test_loader, loss_function, epoch=epoch, dev=False)
            net = prune_architecture(net, top1_acc, prune_agent)
        elif prune_agent.strategy == "weight_sharing":
            top1_acc, _, _ = eval_network(net, test_loader, loss_function, epoch=epoch, dev=False)
            net = prune_architecture(net, top1_acc, prune_agent)
    
    if epoch == settings.COMPRESSION_EPOCH + 1:
        free_initial_weight(net)
        torch.save(net, f'models/{args.net}_{args.dataset}_Out_of_Time_{start_time}.pth')

    end_time = int(time.time())
    logging.info(f'Compress process takes {(end_time - start_time) / 60} minutes')
