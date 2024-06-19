import os
import time
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from thop import profile
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
import logging

from conf import settings
from utils import (Custom_Conv2d, Custom_Linear, count_custom_conv2d, count_custom_linear, get_net_class, 
                   get_dataloader, setup_logging, Prune_agent, torch_set_random_seed)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")


def fine_tuning_network_knowledge_distillation(teacher_net: nn.Module,
                        student_net: nn.Module,
                        target_optimizer: optim.Optimizer, 
                        target_train_loader: torch.utils.data.DataLoader, 
                        loss_function: nn.Module, 
                        epoch: int, 
                        T: float = 2,
                        soft_loss_weight: float = 0.25, 
                        stu_loss_weight: float = 0.75,
                        fft: bool = False):
    
    teacher_net.eval()
    student_net.train()
    train_loss = 0.0
    if fft == True:
        description = "Fine tuning final architecture"
        max_epoch = settings.C_FT_EPOCH
    else:
        description = "Training best generated Architecture"
        max_epoch = settings.C_DEV_NUM
    with tqdm(total=len(target_train_loader), desc=f'{description}: Epoch {epoch}/{max_epoch}', unit='batch', leave=False) as pbar:
        for _, (images, labels) in enumerate(target_train_loader):
            images, labels = images.to(device), labels.to(device)
            
            target_optimizer.zero_grad()
            stu_outputs = student_net(images)
            stu_loss = loss_function(stu_outputs, labels)

            with torch.no_grad():
                tch_outputs = teacher_net(images)
            # soften the student output by applying softmax firstly and log() secondly to avoid overflow and improve efficiency, and teacher output softmax only
            stu_outputs_softened = nn.functional.log_softmax(stu_outputs / T, dim=-1)
            tch_outputs_softened = nn.functional.softmax(tch_outputs / T, dim=-1)
            soft_loss = torch.sum(tch_outputs_softened * (tch_outputs_softened.log() - stu_outputs_softened)) / stu_outputs_softened.shape[0] * T ** 2

            loss = soft_loss_weight * soft_loss + stu_loss_weight * stu_loss
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
    FT_optimizer = optim.SGD(best_new_net.parameters(), lr=settings.T_FT_LR_SCHEDULAR_INITIAL_LR, momentum=0.9, weight_decay=5e-4)
    FT_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(FT_optimizer, settings.C_DEV_NUM - 5, eta_min=settings.T_LR_SCHEDULAR_MIN_LR,last_epoch=-1)
    best_acc = 0.0
    for dev_epoch in range(1, settings.C_DEV_NUM + 1):
        train_loss = fine_tuning_network_knowledge_distillation(teacher_net=teacher_net, 
                                                                student_net=best_new_net, 
                                                                target_optimizer=FT_optimizer, 
                                                                target_train_loader=train_loader, 
                                                                loss_function=loss_function, 
                                                                epoch=dev_epoch, 
                                                                fft=False)
        top1_acc, top5_acc, _ = eval_network(target_net=best_new_net, target_eval_loader=test_loader, loss_function=loss_function)
        logging.info(f"epoch: {dev_epoch}/{settings.C_DEV_NUM}, train_Loss: {train_loss}, top1_acc: {top1_acc}, top5_acc: {top5_acc}")
        FT_lr_scheduler.step()

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
    tolerance_time = settings.RL_LR_TOLERANCE
    for _ in range(1, prune_agent.lr_epoch + 1):
        # we only use the last updated net_list and Q_value_dict for fine tuning, and previous trajectory is used for updating prune distribution
        Q_value_dict = {}
        sample_trajectory(cur_step=0,
                        original_net=original_net,
                        prune_agent=prune_agent,
                        Q_value_dict=Q_value_dict,
                        prev_prune_counter_sum=0)
        logging.info(f'Generated net Q value List: {Q_value_dict[0]}')
        logging.info(f'Current new net Q value cache: {prune_agent.ReplayBuffer[:, 0]}')
        logging.info(f'Current prune probability distribution cache: {prune_agent.ReplayBuffer[:, 1:]}')
        logging.info(f'Previous Q value max: {prune_agent.cur_Q_value_max}')
        logging.info(f'Current Q value max: {torch.max(Q_value_dict[0])}')
        # only update distribution when sampled trajectory is better
        if (torch.max(Q_value_dict[0]) - prune_agent.cur_Q_value_max) / prune_agent.cur_Q_value_max <= settings.RL_REWARD_CHANGE_THRESHOLD and prune_agent.cur_Q_value_max != -1:
            tolerance_time -= 1
            if tolerance_time <= 0:
                break
        else:
            prune_agent.cur_Q_value_max = torch.max(Q_value_dict[0])
            prune_distribution_change = prune_agent.update_prune_distribution(settings.RL_STEP_LENGTH, settings.RL_PROBABILITY_LOWER_BOUND, settings.RL_PPO_CLIP, settings.RL_PPO_ENABLE)
            logging.info(f"current prune probability distribution change: {prune_distribution_change}")
            logging.info(f"current prune probability distribution: {prune_agent.prune_distribution}")
            tolerance_time = settings.RL_LR_TOLERANCE
    
    # use epsilon-greedy exploration strategy
    if torch.rand(1).item() < settings.RL_GREEDY_EPSILON:
        best_net_index = torch.randint(0, settings.RL_MAX_GENERATE_NUM, (1,)).item()
        logging.info(f'Exploration: Net {best_net_index} is the best new net')
    else:
        best_net_index = torch.argmax(prune_agent.ReplayBuffer[:, 0])
        logging.info(f'Exploitation: Net {best_net_index} is the best new net')
    best_generated_net = prune_agent.net_list[best_net_index]
    if wandb_available:
        wandb.log({"optimal_net_reward": prune_agent.ReplayBuffer[best_net_index, 0]}, step=epoch)
    
    return best_generated_net

def sample_trajectory(cur_step: int, 
                      original_net: nn.Module, 
                      prune_agent: Prune_agent, 
                      Q_value_dict: dict, 
                      prev_prune_counter_sum: int):
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
            sample_trajectory(cur_step + 1, generated_net, prune_agent, Q_value_dict, cur_prune_counter_sum)

            if cur_step + 1 in Q_value_dict:
                Q_value_dict[cur_step][model_id] += settings.RL_DISCOUNT_FACTOR * torch.max(Q_value_dict[cur_step + 1])
            
            # update Q_value and ReplayBuffer at top level
            if cur_step == 0:
                min_top1_acc, min_idx = torch.min(prune_agent.ReplayBuffer[:, 0], dim=0)
                if Q_value_dict[0][model_id] >= min_top1_acc:
                    prune_agent.ReplayBuffer[min_idx, 0] = Q_value_dict[0][model_id]
                    prune_agent.ReplayBuffer[min_idx, 1:] = prune_distribution_action
                    prune_agent.net_list[min_idx] = generated_net
            
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

    if wandb_available:
        wandb.log({"generated_net_top1_acc": new_net_top1_acc}, step=epoch)
        wandb.log({"optimal_net_index": optimal_net_index}, step=epoch)
    logging.info(f'Generated Model Top1 Accuracy List: {[original_net_top1_acc, new_net_top1_acc]}, Top5 Accuracy List: {[original_net_top5_acc, new_net_top5_acc]}')
    return optimal_net, optimal_net_index


def get_args():
    parser = argparse.ArgumentParser(description='Adaptively compress the given trained model under given dataset')
    parser.add_argument('-n', '--net', type=str, default=None, help='the type of model to train')
    parser.add_argument('-nid', '--net_id', type=str, default=None, help='the id specific which orginal model to be compressed')
    parser.add_argument('-d', '--dataset', type=str, default=None, help='the dataset to train on')
    parser.add_argument('-r', '--resume', action='store_true', default=False, help='resume the previous target compression')
    parser.add_argument('-reid', '--resume_id', type=int, default=None, help='the id specific previous compression that to be resumed')
    parser.add_argument('-raid', '--random_seed', type=int, default=None, help='the random seed for the current new compression')

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args: argparse.Namespace):
    if args.net is None:
        raise TypeError(f"the specific type of model should be provided, please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
    elif args.net not in ['lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet']:
        raise TypeError(f"the specific model {args.net} is not supported, please specify which original model to be compressed")
    if args.net_id is None and args.resume == False:
        raise TypeError(f"the specific model {args.net_id} should be provided, please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
    if args.dataset is None:
        raise TypeError(f"the specific type of dataset to train on should be provided, please select one of 'mnist', 'cifar10', 'cifar100', 'imagenet'")
    elif args.dataset not in ['mnist', 'cifar10', 'cifar100', 'imagenet']:
        raise TypeError(f"the specific dataset {args.dataset} is not supported, please select one of 'mnist', 'cifar10', 'cifar100', 'imagenet'")
    if args.resume == True and args.resume_id is None:
        raise TypeError(f"the specific resume_id {args.resume_id} should be provided, please specify which compression to resume")


if __name__ == '__main__':
    args = get_args()
    if args.resume == True:
        random_seed = args.resume_id
    elif args.random_seed is not None:
        random_seed = args.resume_id
    else:
        random_seed = int(time.time())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    setup_logging(experiment_id=random_seed, net=args.net, dataset=args.dataset, action='compress')
    
    # reinitialize random seed
    torch_set_random_seed(random_seed)
    logging.info(f'Start with random seed: {random_seed}')

    train_loader, valid_loader, test_loader, _, _ = get_dataloader(dataset=args.dataset, pin_memory=True)
    net_class = get_net_class(net=args.net)
    
    # initialize training parameter
    loss_function = nn.CrossEntropyLoss()

    # initialize parameter to compute complexity of model
    if args.net == 'lenet5':
        input = torch.rand(1, 1, 32, 32).to(device)
    else:
        input = torch.rand(1, 3, 32, 32).to(device)
    custom_ops = {Custom_Conv2d: count_custom_conv2d, Custom_Linear: count_custom_linear}


    prev_epoch = 0
    prev_reached_final_fine_tuning = False
    prev_checkpoint = None

    if args.resume:
        net = torch.load(f'models/{args.net}_{args.dataset}_{args.resume_id}_temp.pth').to(device)
        teacher_net = torch.load(f'models/{args.net}_{args.dataset}_{args.resume_id}_teacher.pth').to(device)

        # resume the previous compression
        prev_checkpoint = torch.load(f"checkpoint/{args.net}_{args.dataset}_{args.resume_id}_checkpoint.pth")
        prev_epoch = prev_checkpoint['epoch']
        prev_reached_final_fine_tuning = prev_checkpoint['reached_final_fine_tuning']
        
        original_para_num = prev_checkpoint['original_para_num']
        original_FLOPs_num = prev_checkpoint['original_FLOPs_num']
        Para_compression_ratio = prev_checkpoint['Para_compression_ratio']
        FLOPs_compression_ratio = prev_checkpoint['FLOPs_compression_ratio']
        
        prune_agent = prev_checkpoint['prune_agent']
        
        initial_protect_used = prev_checkpoint['initial_protect_used']
        best_acc = prev_checkpoint['best_acc']
        initial_top1_acc = prev_checkpoint['initial_top1_acc']
        cur_top1_acc = prev_checkpoint['cur_top1_acc']
    else:
        net = torch.load(f'models/{args.net}_{args.dataset}_{args.net_id}_original.pth').to(device)
        teacher_net = copy.deepcopy(net).to(device)
        torch.save(teacher_net, f'models/{args.net}_{args.dataset}_{random_seed}_teacher.pth')

        # get complexity of original model
        original_FLOPs_num, original_para_num = profile(model=net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
        Para_compression_ratio = 0.0
        FLOPs_compression_ratio = 0.0

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
        best_acc = 0
        initial_top1_acc, _, _ = eval_network(target_net=net, target_eval_loader=test_loader, loss_function=loss_function)
        cur_top1_acc = initial_top1_acc
        if wandb_available:
            wandb.log({"top1_acc": cur_top1_acc, "modification_num": prune_agent.modification_num, "FLOPs_compression_ratio": FLOPs_compression_ratio, "Para_compression_ratio": Para_compression_ratio}, step=0)
            for i in range(net.prune_choices_num):
                wandb.log({f"prune_distribution_item_{i}": prune_agent.prune_distribution[i]}, step=0)

    
    for epoch in range(1, settings.C_COMPRESSION_EPOCH + 1):
        if prev_reached_final_fine_tuning == True or epoch <= prev_epoch:
            continue

        net = prune_architecture(net=net, prune_agent=prune_agent, epoch=epoch)

        if cur_top1_acc >= initial_top1_acc - settings.C_OVERALL_ACCURACY_CHANGE_THRESHOLD:
            torch.save(net, f'models/{args.net}_{args.dataset}_{random_seed}_compressed.pth')
        if wandb_available:
            wandb.log({"top1_acc": cur_top1_acc, "modification_num": prune_agent.modification_num, "FLOPs_compression_ratio": FLOPs_compression_ratio, "Para_compression_ratio": Para_compression_ratio}, step=epoch)
            for i in range(net.prune_choices_num):
                wandb.log({f"prune_distribution_item_{i}": prune_agent.prune_distribution[i]}, step=epoch)
        logging.info(f'Epoch: {epoch}/{settings.C_COMPRESSION_EPOCH}, modification_num: {prune_agent.modification_num}, compression ratio: FLOPs: {FLOPs_compression_ratio}, Parameter number {Para_compression_ratio}')

        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'reached_final_fine_tuning': False,
            'prune_agent': prune_agent,
            'original_para_num': original_para_num,
            'original_FLOPs_num': original_FLOPs_num,
            'Para_compression_ratio': Para_compression_ratio,
            'FLOPs_compression_ratio': FLOPs_compression_ratio,
            'initial_protect_used': initial_protect_used,
            'best_acc': best_acc,
            'initial_top1_acc': initial_top1_acc,
            'cur_top1_acc': cur_top1_acc,
            'FFT_optimizer_state_dict': None,
            'FFT_lr_scheduler_state_dict': None
        }
        torch.save(net, f'models/{args.net}_{args.dataset}_{random_seed}_temp.pth')
        torch.save(checkpoint, f"checkpoint/{args.net}_{args.dataset}_{random_seed}_checkpoint.pth")
    

    FFT_optimizer = optim.SGD(net.parameters(), lr=settings.T_FT_LR_SCHEDULAR_INITIAL_LR, momentum=0.9, weight_decay=5e-4)
    FFT_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(FFT_optimizer, settings.C_FT_EPOCH - 10, eta_min=settings.T_LR_SCHEDULAR_MIN_LR,last_epoch=-1)
    cur_top1_acc = 0
    if args.resume:
        if prev_checkpoint['FFT_optimizer_state_dict'] is not None:
            FFT_optimizer.load_state_dict(prev_checkpoint['FFT_optimizer_state_dict'])
            FFT_lr_scheduler.load_state_dict(prev_checkpoint['FFT_lr_scheduler_state_dict'])
            cur_top1_acc = prev_checkpoint['cur_top1_acc']
    
    for epoch in range(1, settings.C_FT_EPOCH + 1):
        if prev_reached_final_fine_tuning == True and epoch <= prev_epoch:
            continue

        train_loss = fine_tuning_network_knowledge_distillation(teacher_net=teacher_net,
                                                                student_net=net,
                                                                target_optimizer=FFT_optimizer,
                                                                target_train_loader=train_loader,
                                                                loss_function=loss_function, 
                                                                epoch=epoch, 
                                                                fft=True)
        cur_top1_acc, _, _ = eval_network(target_net=net,target_eval_loader=test_loader,loss_function=loss_function)
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss},Top1 Accuracy: {cur_top1_acc}')
        if wandb_available:
            wandb.log({"overall_fine_tuning_train_loss": train_loss,"overall_fine_tuning_top1_acc": cur_top1_acc}, step=epoch+settings.C_COMPRESSION_EPOCH)

        FFT_lr_scheduler.step()

        # start to save best performance model after first training milestone
        if best_acc < cur_top1_acc:
            best_acc = cur_top1_acc

            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, f'models/{args.net}_{args.dataset}_{random_seed}_finished.pth')
        
        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'reached_final_fine_tuning': True,
            'prune_agent': prune_agent,
            'original_para_num': original_para_num,
            'original_FLOPs_num': original_FLOPs_num,
            'Para_compression_ratio': Para_compression_ratio,
            'FLOPs_compression_ratio': FLOPs_compression_ratio,
            'initial_protect_used': initial_protect_used,
            'best_acc': best_acc,
            'initial_top1_acc': initial_top1_acc,
            'cur_top1_acc': cur_top1_acc,
            'FFT_optimizer_state_dict': FFT_optimizer.state_dict(),
            'FFT_lr_scheduler_state_dict': FFT_lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, f"checkpoint/{args.net}_{args.dataset}_{random_seed}_checkpoint.pth")
    
    if wandb_available:
        wandb.finish()
