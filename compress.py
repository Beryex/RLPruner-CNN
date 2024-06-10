import os
import time
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import logging
import copy
from thop import profile

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
          epoch: int, 
          final_ft: bool):
    
    target_net.train()
    train_loss = 0.0
    if final_ft == True:
        description = "Fine tuning final architecture"
        max_epoch = settings.C_FT_EPOCH
    else:
        description = "Training best generated Architecture"
        max_epoch = settings.C_DEV_NUM
    with tqdm(total=len(target_train_loader), desc=f'{description}: Epoch {epoch}/{max_epoch}', unit='batch', leave=False) as pbar:
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
                 loss_function: nn.Module):
    
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
    return top1_acc, top5_acc, eval_loss


def prune_architecture(net: nn.Module, 
                       prune_agent: Prune_agent, 
                       epoch: int):
    
    net, optimal_net_index = get_optimal_architecture(original_net=net, prune_agent=prune_agent)
    prune_agent.step(optimal_net_index, epoch)
    
    return net

def get_optimal_architecture(original_net: nn.Module, 
                             prune_agent: Prune_agent):
    # generate architecture
    best_new_net = get_best_generated_architecture(original_net=original_net, prune_agent=prune_agent)
    
    torch.save(best_new_net, 'models/test_net.pth')
    # fine tuning best new architecture
    optimizer = optim.SGD(best_new_net.parameters(), lr=settings.T_FT_LR_SCHEDULAR_INITIAL_LR, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.C_DEV_NUM - 5, eta_min=settings.T_LR_SCHEDULAR_MIN_LR,last_epoch=-1)
    best_acc = 0.0
    for dev_epoch in range(1, settings.C_DEV_NUM + 1):
        train_loss = fine_tuning_network(target_net=best_new_net, target_optimizer=optimizer, target_train_loader=train_loader, loss_function=loss_function, epoch=dev_epoch, final_ft=False)
        top1_acc, top5_acc, _ = eval_network(target_net=best_new_net, target_eval_loader=test_loader, loss_function=loss_function)
        logging.info(f"epoch: {dev_epoch}/{settings.C_DEV_NUM}, train_Loss: {train_loss}, top1_acc: {top1_acc}, top5_acc: {top5_acc}")
        lr_scheduler.step()

        if best_acc < top1_acc:
            best_acc = top1_acc
            torch.save(best_new_net, f'models/temporary_net.pth')
    best_new_net = torch.load('models/temporary_net.pth').to(device)
    
    # compare best_new_net with original net
    optimal_net, optimal_net_index = evaluate_best_new_net(original_net=original_net, best_new_net=best_new_net, target_eval_loader=test_loader, prune_agent=prune_agent)
    
    if optimal_net_index == 0:
        optimal_net_FLOPs, optimal_net_Params = profile(model=original_net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    else:
        optimal_net_FLOPs, optimal_net_Params = profile(model=best_new_net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    
    global FLOPs_compression_ratio
    global Para_compression_ratio
    FLOPs_compression_ratio = 1 - optimal_net_FLOPs / original_FLOPs_num
    Para_compression_ratio = 1 - optimal_net_Params / original_para_num

    
    return optimal_net, optimal_net_index

def get_best_generated_architecture(original_net: nn.Module,
                                    prune_agent: Prune_agent):
    for _ in range(settings.RL_LR_EPOCH):
        # we only use the last updated net_list and Q_value_dict for fine tuning, and previous trajectory is used for updating prune distribution
        net_list = []
        Q_value_dict = {}
        sample_trajectory(cur_step=0,
                        original_net=original_net,
                        prune_agent=prune_agent,
                        Q_value_dict=Q_value_dict,
                        prev_prune_counter_sum=0,
                        net_list=net_list)
        prune_distribution_change = prune_agent.update_prune_distribution(settings.RL_STEP_LENGTH, settings.RL_PROBABILITY_LOWER_BOUND, settings.RL_PPO_CLIP, settings.RL_PPO_ENABLE)
        logging.info(f"current prune probability distribution: {prune_agent.prune_distribution}")
        logging.info(f"current prune probability distribution change: {prune_distribution_change}")
    
    # use epsilon-greedy exploration strategy
    if torch.rand(1).item() < settings.RL_GREEDY_EPSILON:
        best_net_index = torch.randint(0, settings.RL_MAX_GENERATE_NUM, (1,)).item()
        logging.info(f'Exploration: Net {best_net_index} is the best new net')
    else:
        best_net_index = torch.argmax(Q_value_dict[0])
        logging.info(f'Exploitation: Net {best_net_index} is the best new net')
    best_generated_net = net_list[best_net_index]
    wandb.log({"optimal_net_reward": Q_value_dict[0][best_net_index]}, step=epoch)
    
    logging.info(f'Generated net Q value List: {Q_value_dict[0]}')
    logging.info(f'Current new net Q value cache: {prune_agent.ReplayBuffer[:, 0]}')
    logging.info(f'Current prune probability distribution cache: {prune_agent.ReplayBuffer[:, 1:]}')
    return best_generated_net

def sample_trajectory(cur_step: int, 
                      original_net: nn.Module, 
                      prune_agent: Prune_agent, 
                      Q_value_dict: dict, 
                      prev_prune_counter_sum: int, 
                      net_list: list):
    # sample trajectory using DFS
    if cur_step == settings.RL_MAX_SAMPLE_STEP:
        return
    
    cur_generate_num = max(settings.RL_MAX_GENERATE_NUM // (settings.RL_GENERATE_NUM_SCALING_FACTOR ** cur_step), 1)
    Q_value_dict[cur_step] = torch.zeros(cur_generate_num)

    with tqdm(total=cur_generate_num, desc=f'Generated architectures', unit='model', leave=False) as pbar:
        for model_id in range(cur_generate_num):
            # generate architecture
            generated_net = copy.deepcopy(original_net).to(device)
            prune_distribution_action, prune_counter = net_class.update_architecture(generated_net, prune_agent, settings.RL_PROBABILITY_LOWER_BOUND)

            # evaluate generated architecture
            cur_prune_counter_sum = prune_counter.int() + prev_prune_counter_sum
            tuple_key = tuple(cur_prune_counter_sum.tolist())
            if tuple_key not in prune_agent.Reward_cache:
                top1_acc, _, _ = eval_network(target_net=generated_net, target_eval_loader=test_loader, loss_function=loss_function)
                Q_value_dict[cur_step][model_id] = top1_acc
                prune_agent.Reward_cache[tuple_key] = top1_acc
            else:
                Q_value_dict[cur_step][model_id] = prune_agent.Reward_cache[tuple_key]
            sample_trajectory(cur_step + 1, generated_net, prune_agent, Q_value_dict, cur_prune_counter_sum, net_list)

            if cur_step + 1 in Q_value_dict:
                Q_value_dict[cur_step][model_id] += settings.RL_DISCOUNT_FACTOR * torch.max(Q_value_dict[cur_step + 1])
            
            # update Q_value and ReplayBuffer at top level
            if cur_step == 0:
                net_list.append(generated_net)

                min_top1_acc, min_idx = torch.min(prune_agent.ReplayBuffer[:, 0], dim=0)
                if Q_value_dict[0][model_id] >= min_top1_acc:
                    prune_agent.ReplayBuffer[min_idx, 0] = Q_value_dict[0][model_id]
                    prune_agent.ReplayBuffer[min_idx, 1:] = prune_distribution_action

            pbar.update(1)


def evaluate_best_new_net(original_net: nn.Module, 
                          best_new_net: nn.Module, 
                          target_eval_loader: torch.utils.data.DataLoader,
                          prune_agent: Prune_agent):
    original_net_top1_acc, original_net_top5_acc, _ = eval_network(target_net=original_net, target_eval_loader=target_eval_loader, loss_function=loss_function)
    new_net_top1_acc, new_net_top5_acc, _ = eval_network(target_net=best_new_net, target_eval_loader=target_eval_loader, loss_function=loss_function)
    
    global initial_protect_used
    global cur_top1_acc
    if initial_protect_used == True:
        initial_protect_used = False
        optimal_net = best_new_net
        optimal_net_index = 1
        cur_top1_acc = new_net_top1_acc
        logging.info('Generated net wins')
    elif (original_net_top1_acc - new_net_top1_acc) / original_net_top1_acc > prune_agent.cur_single_step_acc_threshold:
        optimal_net = original_net
        optimal_net_index = 0
        cur_top1_acc = original_net_top1_acc
        logging.info('Original net wins')
    else:
        optimal_net = best_new_net
        optimal_net_index = 1
        cur_top1_acc = new_net_top1_acc
        logging.info('Generated net wins')

    wandb.log({"generated_net_top1_acc": new_net_top1_acc}, step=epoch)
    wandb.log({"optimal_net_index": optimal_net_index}, step=epoch)
    logging.info(f'Generated Model Top1 Accuracy List: {[original_net_top1_acc, new_net_top1_acc]}, Top5 Accuracy List: {[original_net_top5_acc, new_net_top5_acc]}')
    return optimal_net, optimal_net_index


def get_args():
    parser = argparse.ArgumentParser(description='Adaptively compress the given trained model under given dataset')
    parser.add_argument('-net', type=str, default=None, help='the type of model to train')
    parser.add_argument('-dataset', type=str, default=None, help='the dataset to train on')

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args: argparse.Namespace):
    if args.net is None:
        raise TypeError(f"the specific type of model should be provided, please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
    elif args.net not in ['lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet']:
        raise TypeError(f"the specific model {args.net} is not supported, please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
    if args.dataset is None:
        raise TypeError(f"the specific type of dataset to train on should be provided, please select one of 'mnist', 'cifar10', 'cifar100', 'imagenet'")
    elif args.dataset not in ['mnist', 'cifar10', 'cifar100', 'imagenet']:
        raise TypeError(f"the specific dataset {args.dataset} is not supported, please select one of 'mnist', 'cifar10', 'cifar100', 'imagenet'")


if __name__ == '__main__':
    start_time = int(time.time())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    setup_logging(experiment_id=start_time, net=args.net, dataset=args.dataset, action='compress')
    
    # reinitialize random seed
    torch_set_random_seed(start_time)
    logging.info(f'Start with random seed: {start_time}')

    # process input arguments
    train_loader, valid_loader, test_loader, _, _ = get_dataloader(dataset=args.dataset, pin_memory=True)
    net_class = get_net_class(net=args.net)
    net = torch.load('models/vgg16_cifar100_Original_1717669735.pth').to(device)  # replace it with the model gained by train.py

    # compute inital data
    if args.net == 'lenet5':
        input = torch.rand(1, 1, 32, 32).to(device)
    else:
        input = torch.rand(1, 3, 32, 32).to(device)
    custom_ops = {Custom_Conv2d: count_custom_conv2d, Custom_Linear: count_custom_linear}
    original_FLOPs_num, original_para_num = profile(model=net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    Para_compression_ratio = 0.0
    FLOPs_compression_ratio = 0.0

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
    ReplayBuffer = torch.zeros([settings.RL_MAX_GENERATE_NUM, 1 + net.prune_choices_num]) # [:, 0] stores Q(s, a), [:, 1:] stores action a
    prune_agent = Prune_agent(prune_distribution=prune_distribution, ReplayBuffer=ReplayBuffer, filter_num=filter_num)
    logging.info(f'Initial prune probability distribution: {prune_agent.prune_distribution}')

    # initialize compressing parameter
    initial_protect_used = True
    initial_top1_acc, _, _ = eval_network(target_net=net, target_eval_loader=test_loader, loss_function=loss_function)
    cur_top1_acc = initial_top1_acc
    wandb.log({"top1_acc": cur_top1_acc, "modification_num": prune_agent.modification_num, "FLOPs_compression_ratio": FLOPs_compression_ratio, "Para_compression_ratio": Para_compression_ratio}, step=0)
    for i in range(net.prune_choices_num):
        wandb.log({f"prune_distribution_item_{i}": prune_agent.prune_distribution[i]}, step=0)

    
    for epoch in range(1, settings.C_COMPRESSION_EPOCH + 1):
        net = prune_architecture(net=net, prune_agent=prune_agent, epoch=epoch)

        if cur_top1_acc >= initial_top1_acc - settings.C_OVERALL_ACCURACY_CHANGE_THRESHOLD:
            torch.save(net, f'models/{args.net}_{args.dataset}_{start_time}_top1acc_{cur_top1_acc}_FLOPsCR_{FLOPs_compression_ratio}_paraCR_{Para_compression_ratio}.pth')
        wandb.log({"top1_acc": cur_top1_acc, "modification_num": prune_agent.modification_num, "FLOPs_compression_ratio": FLOPs_compression_ratio, "Para_compression_ratio": Para_compression_ratio}, step=epoch)
        for i in range(net.prune_choices_num):
            wandb.log({f"prune_distribution_item_{i}": prune_agent.prune_distribution[i]}, step=epoch)
        logging.info(f'Epoch: {epoch}/{settings.C_COMPRESSION_EPOCH}, modification_num: {prune_agent.modification_num}, compression ratio: FLOPs: {FLOPs_compression_ratio}, Parameter number {Para_compression_ratio}')
    
    optimizer = optim.SGD(net.parameters(), lr=settings.T_FT_LR_SCHEDULAR_INITIAL_LR, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.C_FT_EPOCH - 10, eta_min=settings.T_LR_SCHEDULAR_MIN_LR,last_epoch=-1)
    best_acc = 0
    cur_top1_acc = 0
    
    for epoch in range(1, settings.C_FT_EPOCH + 1):
        train_loss = fine_tuning_network(target_net=net,target_optimizer=optimizer,target_train_loader=train_loader,loss_function=loss_function, epoch=epoch, final_ft=True)
        cur_top1_acc, _, _ = eval_network(target_net=net,target_eval_loader=test_loader,loss_function=loss_function)
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss},Top1 Accuracy: {cur_top1_acc}')
        wandb.log({"overall_fine_tuning_train_loss": train_loss,"overall_fine_tuning_top1_acc": cur_top1_acc}, step=epoch+settings.C_COMPRESSION_EPOCH)

        lr_scheduler.step()

        # start to save best performance model after first training milestone
        if best_acc < cur_top1_acc:
            best_acc = cur_top1_acc

            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, f'models/{args.net}_{args.dataset}_{start_time}_finished.pth')
    
    end_time = int(time.time())
    logging.info(f'Compress process takes {(end_time - start_time) / 60} minutes')
