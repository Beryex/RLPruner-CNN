import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from thop import profile
import random
import numpy as np
from typing import Tuple, List
import wandb
import logging

from conf import settings
from utils import (extract_prunable_layers_info, extract_prunable_layer_dependence, 
                   adjust_prune_distribution_for_cluster, get_dataloader, setup_logging, 
                   Prune_agent, torch_set_random_seed, torch_resume_random_seed)


def main():
    global args
    global device
    global train_loader
    global eval_loader
    global loss_function
    global initial_protect_used
    global cur_top1_acc
    args = get_args()
    device = args.device

    """ Setup logging and get model, data loader, loss function """
    if args.resume:
        prev_checkpoint = torch.load(f"checkpoint/{args.resume_id}_checkpoint.pth")
        random_seed = prev_checkpoint['random_seed']
        experiment_id = args.resume_id

        prev_epoch = prev_checkpoint['epoch']

        model_name = prev_checkpoint['model_name']
        dataset_name = prev_checkpoint['dataset_name']
        setup_logging(experiment_id, model_name, dataset_name, action='compress')
        logging.info(f'Resume Logging setup complete for experiment id: {experiment_id}')

        model = torch.load(f'models/{experiment_id}_checkpoint.pth').to(device)
        teacher_id = prev_checkpoint['teacher_id']
        teacher_model = torch.load(f'models/{model_name}_{dataset_name}_{teacher_id}_original.pth').to(device)
        train_loader, valid_loader, test_loader, _, _ = get_dataloader(args.dataset, 
                                                                       batch_size=args.batch_size, 
                                                                       num_workers=args.num_worker)
        eval_loader = test_loader
        loss_function = prev_checkpoint['loss_function']
    else:
        if args.random_seed is not None:
            random_seed = args.random_seed
            experiment_id = int(time.time())
        else:
            random_seed = int(time.time())
            experiment_id = random_seed
        
        prev_epoch = 0

        model_name = args.model
        dataset_name = args.dataset
        setup_logging(experiment_id, model_name, dataset_name, action='compress')
        logging.info(f'Logging setup complete for experiment id: {experiment_id}')

        model = torch.load(f'models/{model_name}_{dataset_name}_{args.model_id}_original.pth').to(device)
        teacher_id = args.model_id
        teacher_model = copy.deepcopy(model).to(device)
        train_loader, valid_loader, test_loader, _, _ = get_dataloader(args.dataset, 
                                                                       batch_size=args.batch_size, 
                                                                       num_workers=args.num_worker)
        eval_loader = test_loader
        loss_function = nn.CrossEntropyLoss()
    
    """ Extract layer dependence for pruning """
    if dataset_name == 'mnist':
        sample_input = torch.rand(1, 1, 32, 32).to(device)
    else:
        sample_input = torch.rand(1, 3, 32, 32).to(device)
    sample_input.requires_grad = True # used to extract dependence

    logging.info(f'Start extracting layers dependency')
    prune_distribution, filter_num, prunable_layers = extract_prunable_layers_info(model)
    next_layers, layer_cluster_mask = extract_prunable_layer_dependence(model, 
                                                                        sample_input, 
                                                                        prunable_layers)
    prune_distribution = adjust_prune_distribution_for_cluster(prune_distribution, 
                                                               layer_cluster_mask)
    model_with_info = (model, prunable_layers, next_layers)
    logging.info(f'Complete extracting layers dependency')
    
    """ get compression data """
    if args.resume:
        original_para_num = prev_checkpoint['original_para_num']
        original_FLOPs_num = prev_checkpoint['original_FLOPs_num']
        Para_compression_ratio = prev_checkpoint['Para_compression_ratio']
        FLOPs_compression_ratio = prev_checkpoint['FLOPs_compression_ratio']

        initial_protect_used = prev_checkpoint['initial_protect_used']
        initial_top1_acc = prev_checkpoint['initial_top1_acc']
        cur_top1_acc = prev_checkpoint['cur_top1_acc']
    else:
        original_FLOPs_num, original_para_num = profile(model=model, 
                                                        inputs = (sample_input, ), 
                                                        verbose=False)
        Para_compression_ratio = 0.0
        FLOPs_compression_ratio = 0.0

        initial_protect_used = True
        initial_top1_acc, _, _ = evaluate(model)
        cur_top1_acc = initial_top1_acc

    """ get prune agent """
    if args.resume:
        prune_agent = prev_checkpoint['prune_agent']
    else:
        # Replay buffer [:, 0] stores Q value Q(s, a), [:, 1:] stores action PD
        ReplayBuffer = torch.zeros([args.sample_num, 1 + len(prune_distribution)])
        prune_agent = Prune_agent(prune_distribution=prune_distribution,
                                  layer_cluster_mask=layer_cluster_mask,
                                  ReplayBuffer=ReplayBuffer, 
                                  filter_num=filter_num, 
                                  cur_top1_acc=cur_top1_acc)
        logging.info(f'Initial prune probability distribution: {prune_agent.prune_distribution}')

        wandb.log({"top1_acc": cur_top1_acc, 
                   "modification_num": prune_agent.modification_num, 
                   "FLOPs_compression_ratio": FLOPs_compression_ratio, 
                   "Para_compression_ratio": Para_compression_ratio}, 
                   step=prev_epoch)
        for i in range(len(prune_distribution)):
            wandb.log({f"prune_distribution_item_{i}": prune_agent.prune_distribution[i]}, 
                      step=prev_epoch)
    
    """ set random seed for reproducible usage """
    if args.resume:
        torch_resume_random_seed(prev_checkpoint)
        logging.info(f'Resume previous random state')
    else:
        torch_set_random_seed(random_seed)
        logging.info(f'Start with random seed: {random_seed}')
    
    """ begin compressing """
    with tqdm(total=args.epoch, desc=f'Compressing', unit='epoch') as pbar:
        for _ in range(prev_epoch):
            pbar.update(1)

        for epoch in range(prev_epoch, args.epoch + 1):
            """ get generated model """
            generated_model_with_info = generate_architecture(model_with_info, 
                                                              prune_agent, 
                                                              epoch=epoch)
            generated_model = generated_model_with_info[0]

            """ fine tuning generated model """
            optimizer = optim.SGD(generated_model.parameters(), 
                                  lr=args.lr, momentum=0.9, 
                                  weight_decay=5e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                args.fine_tune_epoch - 5, 
                                                                eta_min=args.min_lr,
                                                                last_epoch=-1)
            best_acc = -1
            best_trained_generated_model_with_info = None
            for dev_epoch in range(1, args.fine_tune_epoch + 1):
                train_loss = fine_tuning_with_KD(teacher_model=teacher_model, 
                                                 student_model=generated_model, 
                                                 optimizer=optimizer,
                                                 soft_loss_weight=1-args.stu_co,
                                                 stu_loss_weight=args.stu_co)
                top1_acc, top5_acc, _ = evaluate(generated_model)
                logging.info(f"epoch: {dev_epoch}/{args.fine_tune_epoch}, "
                             f"train_Loss: {train_loss}, "
                             f"top1_acc: {top1_acc}, "
                             f"top5_acc: {top5_acc}")
                lr_scheduler.step()

                if best_acc < top1_acc:
                    best_acc = top1_acc
                    torch.save(generated_model, f"models/{experiment_id}_checkpoint.pth")
                    best_trained_generated_model_with_info = copy.deepcopy(generated_model_with_info)
            
            """ Compare best trained generated model with original one """
            optimal_model_with_info, optimal_model_index = evaluate_best_generated_model(model_with_info, 
                                                                                     best_trained_generated_model_with_info,
                                                                                     prune_agent,
                                                                                     epoch)
            model_with_info = optimal_model_with_info
            optimal_model = optimal_model_with_info[0]
            optimal_model_FLOPs, optimal_model_Params = profile(model=optimal_model, 
                                                            inputs = (sample_input, ), 
                                                            verbose=False)
            FLOPs_compression_ratio = 1 - optimal_model_FLOPs / original_FLOPs_num
            Para_compression_ratio = 1 - optimal_model_Params / original_para_num

            wandb.log({"top1_acc": cur_top1_acc, 
                    "modification_num": prune_agent.modification_num, 
                    "FLOPs_compression_ratio": FLOPs_compression_ratio, 
                    "Para_compression_ratio": Para_compression_ratio}, 
                    step=epoch)
            for i in range(len(prune_agent.prune_distribution)):
                wandb.log({f"prune_distribution_item_{i}": prune_agent.prune_distribution[i]}, 
                        step=epoch)
            logging.info(f'Epoch: {epoch}/{args.epoch}, ' 
                         f'modification_num: {prune_agent.modification_num}, '
                         f'compression ratio: FLOPs: {FLOPs_compression_ratio}, '
                         f'Parameter number {Para_compression_ratio}')
            
            prune_agent.step(optimal_model_index, 
                             epoch,
                             cur_top1_acc)
            pbar.set_postfix({'Para': Para_compression_ratio, 
                              'FLOPs': FLOPs_compression_ratio, 
                              'Top1 acc': best_acc})
            pbar.update(1)

            # save checkpoint
            checkpoint = {
                # compression parameter
                'epoch': epoch,
                'reached_final_fine_tuning': False,
                'model_name': model_name,
                'teacher_id': teacher_id,
                'dataset_name': dataset_name,
                'prune_agent': prune_agent,
                'original_para_num': original_para_num,
                'original_FLOPs_num': original_FLOPs_num,
                'Para_compression_ratio': Para_compression_ratio,
                'FLOPs_compression_ratio': FLOPs_compression_ratio,
                'initial_protect_used': initial_protect_used,
                'initial_top1_acc': initial_top1_acc,
                'cur_top1_acc': cur_top1_acc,

                # random seed parameter
                'random_seed': random_seed,
                'python_hash_seed': os.environ['PYTHONHASHSEED'],
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'cuda_random_state': torch.cuda.get_rng_state_all()
            }
            torch.save(checkpoint, f"checkpoint/{experiment_id}_checkpoint.pt")
            torch.save(optimal_model, f"models/{experiment_id}_checkpoint.pth")


@torch.no_grad()
def evaluate(model: nn.Module):
    """ Evaluate model on eval_loader """
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    eval_loss = 0.0
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        eval_loss += loss_function(outputs, labels).item()
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(eval_loader.dataset)
    top5_acc = correct_5 / len(eval_loader.dataset)
    eval_loss /= len(eval_loader)
    return top1_acc, top5_acc, eval_loss


def generate_architecture(original_model_with_info: Tuple,
                          prune_agent: Prune_agent,
                          epoch: int) -> Tuple[nn.Module, List, List]:
    """ Generate architecture using RL """
    tolerance_time = prune_agent.lr_tolerance_time
    for _ in range(1, prune_agent.lr_epoch + 1):
        Q_value_dict = {}
        sample_trajectory(0,
                          original_model_with_info,
                          prune_agent,
                          Q_value_dict)
        logging.info(f'Generated model Q value List: {Q_value_dict[0]}')
        logging.info(f'Current new model Q value cache: {prune_agent.ReplayBuffer[:, 0]}')
        logging.info(f'Current prune probability distribution cache: {prune_agent.ReplayBuffer[:, 1:]}')
        logging.info(f'Previous Q value max: {prune_agent.cur_Q_value_max}')
        logging.info(f'Current Q value max: {torch.max(Q_value_dict[0])}')
        # only update distribution when sampled trajectory is better
        if ((torch.max(Q_value_dict[0]) - prune_agent.cur_Q_value_max) / prune_agent.cur_Q_value_max
            <= prune_agent.lr_change_threshold):
            tolerance_time -= 1
            if tolerance_time <= 0:
                break
        else:
            prune_agent.cur_Q_value_max = torch.max(Q_value_dict[0])
            prune_distribution_change = prune_agent.update_prune_distribution(args.step_length,
                                                                              args.ppo_clip,
                                                                              args.ppo)
            logging.info(f"current prune probability distribution change: {prune_distribution_change}")
            logging.info(f"current prune probability distribution: {prune_agent.prune_distribution}")
            tolerance_time = prune_agent.lr_tolerance_time
    
    # use epsilon-greedy exploration strategy
    if torch.rand(1).item() < args.greedy_epsilon:
        best_model_index = torch.randint(0, args.sample_num, (1,)).item()
        logging.info(f'Exploration: model {best_model_index} is the best new model')
    else:
        best_model_index = torch.argmax(prune_agent.ReplayBuffer[:, 0])
        logging.info(f'Exploitation: model {best_model_index} is the best new model')
    best_generated_model_with_info = prune_agent.model_info_list[best_model_index]
    wandb.log({"optimal_model_reward": prune_agent.ReplayBuffer[best_model_index, 0]}, step=epoch)
    
    return best_generated_model_with_info

def sample_trajectory(cur_step: int, 
                      original_model_with_info: Tuple,
                      prune_agent: Prune_agent, 
                      Q_value_dict: dict) -> None:
    """ Sample trajectory using DFS """
    if cur_step == args.sample_step:
        return
    
    cur_generate_num = args.sample_num
    Q_value_dict[cur_step] = torch.zeros(cur_generate_num)

    with tqdm(total=cur_generate_num, desc=f'Generated architectures', unit='model', leave=False) as pbar:
        for model_id in range(cur_generate_num):
            generated_model_with_info = copy.deepcopy(original_model_with_info)
            generated_model, generated_prunable_layers, generated_next_layers = generated_model_with_info
            prune_distribution_action = prune_agent.prune_architecture(generated_prunable_layers, 
                                                                       generated_next_layers)

            # evaluate generated architecture
            top1_acc, _, _ = evaluate(generated_model)
            Q_value_dict[cur_step][model_id] = top1_acc
        
            sample_trajectory(cur_step=cur_step + 1, 
                              original_model_with_info=generated_model_with_info, 
                              prune_agent=prune_agent, 
                              Q_value_dict=Q_value_dict)

            if cur_step + 1 in Q_value_dict:
                Q_value_dict[cur_step][model_id] += args.discount_factor * torch.max(Q_value_dict[cur_step + 1])
            
            # update Q_value and ReplayBuffer at top level
            if cur_step == 0:
                min_top1_acc, min_idx = torch.min(prune_agent.ReplayBuffer[:, 0], dim=0)
                if Q_value_dict[0][model_id] >= min_top1_acc:
                    prune_agent.ReplayBuffer[min_idx, 0] = Q_value_dict[0][model_id]
                    prune_agent.ReplayBuffer[min_idx, 1:] = prune_distribution_action
                    prune_agent.model_info_list[min_idx] = generated_model_with_info
            
            pbar.update(1)


def fine_tuning_with_KD(teacher_model: nn.Module,
                        student_model: nn.Module,
                        optimizer: optim.Optimizer, 
                        T: float = 2,
                        soft_loss_weight: float = 0.75, 
                        stu_loss_weight: float = 0.25) -> float:
    """ fine tuning generated model with knowledge distillation with original model as teach """
    teacher_model.eval()
    student_model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        stu_outputs = student_model(images)
        stu_loss = loss_function(stu_outputs, labels)

        with torch.no_grad():
            tch_outputs = teacher_model(images)
        # soften the student output by applying softmax firstly and log() 
        # secondly to avoid overflow and improve efficiency, and teacher output softmax only
        stu_outputs_softened = nn.functional.log_softmax(stu_outputs / T, dim=-1)
        tch_outputs_softened = nn.functional.softmax(tch_outputs / T, dim=-1)
        soft_loss = (torch.sum(tch_outputs_softened * (tch_outputs_softened.log() - stu_outputs_softened)) 
                     / stu_outputs_softened.shape[0] * T ** 2)

        loss = soft_loss_weight * soft_loss + stu_loss_weight * stu_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss


def evaluate_best_generated_model(original_model_with_info: Tuple, 
                                  best_generated_model_with_info: Tuple,
                                  prune_agent: Prune_agent,
                                  epoch: int):
    """ compare the generated model with original one """
    original_model_top1_acc, original_model_top5_acc, _ = evaluate(original_model_with_info[0])
    new_model_top1_acc, new_model_top5_acc, _ = evaluate(best_generated_model_with_info[0])
    
    global initial_protect_used
    global cur_top1_acc
    if (initial_protect_used == True or 
        (original_model_top1_acc - new_model_top1_acc) / original_model_top1_acc < prune_agent.cur_single_step_acc_threshold):
        initial_protect_used = False
        optimal_model_with_info = best_generated_model_with_info
        optimal_model_index = 1
        cur_top1_acc = new_model_top1_acc
        logging.info('Generated model wins')
    else:
        optimal_model_with_info = original_model_with_info
        optimal_model_index = 0
        cur_top1_acc = original_model_top1_acc
        logging.info('Original model wins')

    if new_model_top1_acc <= 0.05:
        logging.info(f'Error Top1 acc: Original model: {original_model_top1_acc[0]}')
        logging.info(f'Error Top1 acc: best generated model: {best_generated_model_with_info[0]}')
    wandb.log({"generated_model_top1_acc": new_model_top1_acc}, step=epoch)
    wandb.log({"optimal_model_index": optimal_model_index}, step=epoch)
    logging.info(f"Generated Model Top1 Accuracy List: {[original_model_top1_acc, new_model_top1_acc]}, "
                 f"Top5 Accuracy List: {[original_model_top5_acc, new_model_top5_acc]}")
    return optimal_model_with_info, optimal_model_index


def get_args():
    parser = argparse.ArgumentParser(description='Compress mode using RLPruner')
    parser.add_argument('--model', '-m', type=str, default=None, 
                        help='the name of model, just used to track logging')
    parser.add_argument('--model-id', '-mid', type=int, default=None, 
                        help='the id specific which model to be compressed')
    parser.add_argument('--dataset', '-ds', type=str, default=None, 
                        help='the dataset to train on')
    parser.add_argument('--noise-var', '-nv', default=settings.RL_PRUNE_FILTER_NOISE_VAR, 
                        help='variance when generating new prune distribution')
    parser.add_argument('--sample_step', '-ss', default=settings.RL_MAX_SAMPLE_STEP, 
                        help='the sample step of prune distribution')
    parser.add_argument('--sample-num', '-sn', default=settings.RL_MAX_SAMPLE_NUM, 
                        help='the sample number of prune distribution')
    parser.add_argument('--discount-factor', '-df', default=settings.RL_DISCOUNT_FACTOR, 
                        help='the discount factor for multi sample step')
    parser.add_argument('--step-length', '-sl', default=settings.RL_STEP_LENGTH, 
                        help='step length when updating prune distribution')
    parser.add_argument('--greedy-epsilon', '-ge', default=settings.RL_GREEDY_EPSILON, 
                        help='the probability to adopt random policy')
    parser.add_argument('--ppo', action='store_true', default=settings.RL_PPO_ENABLE, 
                        help='enable Proximal Policy Optimization')
    parser.add_argument('--ppo-clip', '-ppoc', default=settings.RL_PPO_CLIP, 
                        help='the clip value for PPO')
    parser.add_argument('--lr', type=float, default=settings.T_FT_LR_SCHEDULAR_INITIAL_LR,
                        help='initial fine tuning learning rate')
    parser.add_argument('--min-lr', type=float, default=settings.T_LR_SCHEDULAR_MIN_LR,
                        help='minimal learning rate')
    parser.add_argument('--fine-tune_epoch', '-fte', type=int, default=settings.C_FT_EPOCH,
                        help='fine tuning epoch for generated model')
    parser.add_argument('--stu-co', '-sc', type=float, default=settings.T_FT_STU_CO,
                        help='the student loss coefficient in knowledge distillation')
    parser.add_argument('--acc-change-threshold', '-act', type=int, default=settings.C_ACC_CHANGE_THRESHOLD,
                        help='the acc change threshold to decide whether adopt generated model')
    parser.add_argument('--batch-size', '-b', type=int, default=settings.T_BATCH_SIZE, 
                        help='batch size for dataloader')
    parser.add_argument('--num-worker', '-n', type=int, default=settings.T_NUM_WORKER, 
                        help='num_workers for dataloader')
    parser.add_argument('--epoch', '-e', type=int, default=settings.C_COMPRESSION_EPOCH, 
                        help='total epoch to train')
    parser.add_argument('--device', '-dev', type=str, default='cpu', 
                        help='device to use')
    parser.add_argument('--random-seed', '-rs', type=int, default=None, 
                        help='the random seed for the current new compression')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='resume the previous target compression')
    parser.add_argument('--resume-id', type=int, default=None, 
                        help='the id specific previous compression that to be resumed')

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args: argparse.Namespace):
    if args.resume == False:
        if args.model_id is None:
            raise TypeError(f"the specific model {args.model_id} should be provided")
        if args.dataset is None:
            raise TypeError(f"the specific type of dataset to train on should be provided, "
                            f"please select one of 'mnist', 'cifar10', 'cifar100'")
        elif args.dataset not in ['mnist', 'cifar10', 'cifar100']:
            raise TypeError(f"the specific dataset {args.dataset} is not supported, "
                            f"please select one of 'mnist', 'cifar10', 'cifar100'")
    elif args.resume_id is None:
        raise TypeError(f"the specific resume_id {args.resume_id} should be provided, "
                        f"please specify which compression to resume")


if __name__ == '__main__':
    main()
