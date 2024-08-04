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
from utils import (setup_logging, get_dataloader, get_dataloader_with_checkpoint,
                   RL_Pruner, torch_set_random_seed, torch_resume_random_seed,
                   WarmUpLR, DATASETS, PRUNE_STRATEGY)


def main():
    global args
    global device
    global train_loader
    global eval_loader
    global loss_function
    args = get_args()
    device = args.device
    model_name = args.model
    dataset_name = args.dataset

    pretrained_pth = f"{args.pretrained_dir}/{model_name}_{dataset_name}_original.pth"
    compressed_pth = f"{args.compressed_dir}/{model_name}_{dataset_name}_{args.sparsity}.pth"


    """ Setup logging and get model, data loader, loss function """
    if args.resume:
        prev_checkpoint = torch.load(f"{args.checkpoint_dir}/{args.resume_epoch}/checkpoint.pt")
        
        random_seed = prev_checkpoint['random_seed']
        experiment_id = prev_checkpoint['experiment_id']
        prev_epoch = prev_checkpoint['epoch']

        setup_logging(log_dir=args.log_dir,
                      experiment_id=experiment_id, 
                      random_seed=random_seed,
                      args=args,
                      model_name=args.model, 
                      dataset_name=args.dataset, 
                      action='compress',
                      project_name=args.project_name,
                      use_wandb=args.use_wandb)
        logging.info(f'Resume Logging setup complete for experiment id: {experiment_id}')
        print(f"Resume Logging setup complete for experiment id: {experiment_id}")

        model = torch.load(f"{args.checkpoint_dir}/{args.resume_epoch}/model.pth").to(device)
        teacher_model = torch.load(f"{args.checkpoint_dir}/{args.resume_epoch}/teacher.pth").to(device)
        train_loader, valid_loader, test_loader = get_dataloader_with_checkpoint(prev_checkpoint, 
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

        setup_logging(log_dir=args.log_dir,
                      experiment_id=experiment_id, 
                      random_seed=random_seed,
                      args=args,
                      model_name=args.model, 
                      dataset_name=args.dataset, 
                      action='compress',
                      project_name=args.project_name,
                      use_wandb=args.use_wandb)
        logging.info(f'Logging setup complete for experiment id: {experiment_id}')
        print(f"Logging setup complete for experiment id: {experiment_id}")

        model = torch.load(f"{pretrained_pth}").to(device)
        teacher_model = copy.deepcopy(model).to(device)
        train_loader, valid_loader, test_loader, _, _ = get_dataloader(args.dataset, 
                                                                       batch_size=args.batch_size, 
                                                                       num_workers=args.num_worker)
        eval_loader = test_loader
        loss_function = nn.CrossEntropyLoss()
    
    """ get compression data """
    if dataset_name == 'mnist':
        sample_input = torch.rand(1, 1, 32, 32).to(device)
    else:
        sample_input = torch.rand(1, 3, 32, 32).to(device)

    if args.resume:
        original_para_num = prev_checkpoint['original_para_num']
        original_FLOPs_num = prev_checkpoint['original_FLOPs_num']
        Para_compression_ratio = prev_checkpoint['Para_compression_ratio']
        FLOPs_compression_ratio = prev_checkpoint['FLOPs_compression_ratio']
        teacher_top1_acc = prev_checkpoint['teacher_top1_acc']
    else:
        original_FLOPs_num, original_para_num = profile(model=model, 
                                                        inputs = (sample_input, ), 
                                                        verbose=False)
        Para_compression_ratio = 0.0
        FLOPs_compression_ratio = 0.0

        teacher_top1_acc, _, _ = evaluate(model)

    """ Initialize prune agent to be RL_Pruner """
    if args.resume:
        prune_agent = prev_checkpoint['prune_agent']
        prune_agent.resume_model(model, sample_input)
    else:
        prune_agent = RL_Pruner(model=model,
                                sample_input=sample_input,
                                sample_num=args.sample_num,
                                prune_filter_ratio=args.prune_filter_ratio,
                                noise_var=args.noise_var)
        logging.info(f'Initial prune probability distribution: {prune_agent.prune_distribution}')

        wandb.log({"top1_acc": teacher_top1_acc, 
                   "modification_num": prune_agent.modification_num, 
                   "FLOPs_compression_ratio": FLOPs_compression_ratio, 
                   "Para_compression_ratio": Para_compression_ratio}, 
                   step=prev_epoch)
    model_with_info = prune_agent.get_linked_model()
    
    """ set random seed for reproducible usage """
    if args.resume:
        torch_resume_random_seed(prev_checkpoint)
        logging.info(f'Resume previous random state')
        print(f"Resume previous random state")
    else:
        torch_set_random_seed(random_seed)
        logging.info(f'Start with random seed: {random_seed}')
        print(f"Start with random seed: {random_seed}")

    """ begin compressing """
    pruning_epoch = round(args.sparsity * 100)
    with tqdm(total=pruning_epoch, desc=f'Compressing', unit='epoch') as pbar:
        pbar.update(prev_epoch)

        for epoch in range(1 + prev_epoch, pruning_epoch + 1):
            """ get generated model """
            generated_model_with_info, best_Q_value = generate_architecture(model_with_info, 
                                                                            prune_agent)
            generated_model = generated_model_with_info[0]
            best_acc, _, _ = evaluate(generated_model)
            model_with_info = copy.deepcopy(generated_model_with_info)

            """ fine tuning generated model """
            optimizer = optim.SGD(generated_model.parameters(), 
                                  lr=args.lr, 
                                  momentum=0.9, 
                                  weight_decay=5e-4)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                args.fine_tune_epoch - 5, 
                                                                eta_min=args.min_lr,
                                                                last_epoch=-1)
            iter_per_epoch = len(train_loader)
            lr_scheduler_warmup = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epoch)
            
            with tqdm(total=args.fine_tune_epoch, desc=f'Fine tuning', unit='epoch', leave=False) as pbar2:
                for ft_epoch in range(1, args.fine_tune_epoch + 1):
                    train_loss = fine_tuning_with_KD(teacher_model=teacher_model, 
                                                     student_model=generated_model, 
                                                     optimizer=optimizer,
                                                     lr_scheduler_warmup=lr_scheduler_warmup,
                                                     T=args.KD_temperature,
                                                     soft_loss_weight=1-args.stu_co,
                                                     stu_loss_weight=args.stu_co,
                                                     ft_epoch=ft_epoch)
                    top1_acc, top5_acc, _ = evaluate(generated_model)
                    logging.info(f"epoch: {ft_epoch}/{args.fine_tune_epoch}, "
                                 f"train_Loss: {train_loss}, "
                                 f"top1_acc: {top1_acc}, "
                                 f"top5_acc: {top5_acc}")
                    
                    if ft_epoch > args.warmup_epoch:
                        lr_scheduler.step()

                    if best_acc < top1_acc:
                        best_acc = top1_acc
                        model_with_info = copy.deepcopy(generated_model_with_info)
                
                    pbar2.set_postfix({'train_loss': train_loss,
                                       'Best_acc': best_acc, 
                                       'cur_acc': top1_acc})
                    pbar2.update(1)

            """ Switch teacher if the generated is better """
            if teacher_top1_acc < best_acc:
                logging.info(f"Switch teacher to the generated one")
                teacher_model = model_with_info[0]
                teacher_top1_acc = best_acc
            
            """ Compute compression results """
            model_FLOPs, model_Params = profile(model=model_with_info[0], 
                                                inputs = (sample_input, ), 
                                                verbose=False)
            FLOPs_compression_ratio = 1 - model_FLOPs / original_FLOPs_num
            Para_compression_ratio = 1 - model_Params / original_para_num

            wandb.log({"top1_acc": best_acc, 
                       "best_Q_value": best_Q_value,
                       "modification_num": prune_agent.modification_num, 
                       "FLOPs_compression_ratio": FLOPs_compression_ratio, 
                       "Para_compression_ratio": Para_compression_ratio}, 
                       step=epoch)
            for i in range(len(prune_agent.prune_distribution)):
                wandb.log({f"prune_distribution_item_{i}": prune_agent.prune_distribution[i]}, 
                        step=epoch)
            logging.info(f'Epoch: {epoch}/{pruning_epoch}, ' 
                         f'top1_acc: {best_acc}, '
                         f'best_Q_value: {best_Q_value}, '
                         f'modification_num: {prune_agent.modification_num}, '
                         f'compression ratio: FLOPs: {FLOPs_compression_ratio}, '
                         f'Parameter number {Para_compression_ratio}')
            
            prune_agent.clear_cache()

            pbar.set_postfix({'Para': Para_compression_ratio, 
                                'FLOPs': FLOPs_compression_ratio, 
                                'Q_value': best_Q_value,
                                'Top1_acc': best_acc})
            pbar.update(1)

            # save checkpoint
            if epoch % 5 == 0:
                checkpoint = {
                    # compression parameter
                    'loss_function': loss_function,
                    'train_loader': train_loader.dataset if train_loader is not None else None,
                    'valid_loader': valid_loader.dataset if valid_loader is not None else None,
                    'test_loader': test_loader.dataset if test_loader is not None else None,
                    'train_sampler': train_loader.sampler if train_loader is not None else None,
                    'valid_sampler': valid_loader.sampler if valid_loader is not None else None,
                    'test_sampler': test_loader.sampler if test_loader is not None else None,
                    'experiment_id': experiment_id,
                    'epoch': epoch,
                    'prune_agent': prune_agent,
                    'original_para_num': original_para_num,
                    'original_FLOPs_num': original_FLOPs_num,
                    'Para_compression_ratio': Para_compression_ratio,
                    'FLOPs_compression_ratio': FLOPs_compression_ratio,
                    'teacher_top1_acc': teacher_top1_acc,

                    # random seed parameter
                    'random_seed': random_seed,
                    'python_hash_seed': os.environ['PYTHONHASHSEED'],
                    'random_state': random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'torch_random_state': torch.get_rng_state(),
                    'cuda_random_state': torch.cuda.get_rng_state_all()
                }
                os.makedirs(f"{args.checkpoint_dir}/{epoch}", exist_ok=True)
                torch.save(checkpoint, f"{args.checkpoint_dir}/{epoch}/checkpoint.pt")
                torch.save(model_with_info[0], f"{args.checkpoint_dir}/{epoch}/model.pth")
                torch.save(teacher_model, f"{args.checkpoint_dir}/{epoch}/teacher.pth")
        
        os.makedirs(f"{args.compressed_dir}", exist_ok=True)
        torch.save(model_with_info[0], f"{compressed_pth}")
        logging.info(f"Compressed model saved at {compressed_pth}")
        print(f"Compressed model saved at {compressed_pth}")
        wandb.finish()


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
                          prune_agent: RL_Pruner) -> Tuple[nn.Module, List, List]:
    """ Generate architecture using RL """
    best_Q_value = 0
    with tqdm(total=args.lr_epoch, desc=f'Sampling', unit='epoch', leave=False) as pbar:
        for rl_epoch in range(1, args.lr_epoch + 1):
            Q_value_dict = {}
            sample_trajectory(0,
                              original_model_with_info,
                              prune_agent,
                              Q_value_dict)
            cur_Q_value = torch.max(Q_value_dict[0]).item()
            logging.info(f'Generated model Q value List: {Q_value_dict[0]}')
            logging.info(f'Current new model Q value cache: {prune_agent.ReplayBuffer[:, 0]}')
            logging.info(f'Current prune probability distribution cache: {prune_agent.ReplayBuffer[:, 1:]}')
            logging.info(f'Q value max: {best_Q_value}')
            logging.info(f'Current Q value: {cur_Q_value}')
            # only update distribution when sampled trajectory is better
            if cur_Q_value > best_Q_value:
                best_Q_value = cur_Q_value
                prune_distribution_change = prune_agent.update_prune_distribution(args.step_length,
                                                                                  args.ppo_clip,
                                                                                  args.ppo)
                logging.info(f"current prune probability distribution change: {prune_distribution_change}")
                logging.info(f"current prune probability distribution: {prune_agent.prune_distribution}")

            pbar.set_postfix({'Best Q value': best_Q_value, 
                              'Q value': cur_Q_value})
            pbar.update(1)
    
    # use epsilon-greedy exploration strategy
    if torch.rand(1).item() < args.greedy_epsilon:
        best_model_index = torch.randint(0, args.sample_num, (1,)).item()
        logging.info(f'Exploration: model {best_model_index} is the best new model')
    else:
        best_model_index = torch.argmax(prune_agent.ReplayBuffer[:, 0])
        logging.info(f'Exploitation: model {best_model_index} is the best new model')
    best_generated_model_with_info = prune_agent.model_info_list[best_model_index]
    
    return best_generated_model_with_info, best_Q_value


def sample_trajectory(cur_step: int, 
                      original_model_with_info: Tuple,
                      prune_agent: RL_Pruner, 
                      Q_value_dict: dict) -> None:
    """ Sample trajectory using Depth First Search """
    if cur_step == args.sample_step:
        return
    
    cur_generate_num = args.sample_num
    Q_value_dict[cur_step] = torch.zeros(cur_generate_num)

    for model_id in range(cur_generate_num):
        generated_model_with_info = copy.deepcopy(original_model_with_info)
        generated_model, generated_prunable_layers, generated_next_layers = generated_model_with_info
        
        """ Link the generated model and prune it """
        prune_agent.link_model(generated_model_with_info)
        prune_distribution_action = prune_agent.prune_architecture(eval_loader,
                                                                   args.prune_strategy)

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


def fine_tuning_with_KD(teacher_model: nn.Module,
                        student_model: nn.Module,
                        optimizer: optim.Optimizer, 
                        lr_scheduler_warmup: WarmUpLR,
                        ft_epoch: int,
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

        if ft_epoch <= args.warmup_epoch:
            lr_scheduler_warmup.step()

    train_loss /= len(train_loader)
    return train_loss


def get_args():
    parser = argparse.ArgumentParser(description='Compress mode using RLPruner')
    parser.add_argument('--model', '-m', type=str, default=None, 
                        help='the name of model, just used to track logging')
    parser.add_argument('--dataset', '-ds', type=str, default=None, 
                        help='the dataset to train on')
    parser.add_argument('--sparsity', '-s', type=float, default=settings.C_SPARSITY, 
                        help='the overall pruning sparsity')
    parser.add_argument('--prune_strategy', '-ps', type=str, default=settings.C_PRUNE_STRATEGY, 
                        help='strategy to evaluate unimportant weights')
    parser.add_argument('--prune_filter_ratio', '-pfr', type=float, default=settings.C_PRUNE_FILTER_RATIO,
                        help='what ratio of filter to prune for each pruning')
    parser.add_argument('--noise_var', '-nv', type=float, default=settings.RL_PRUNE_FILTER_NOISE_VAR, 
                        help='variance when generating new prune distribution')
    parser.add_argument('--lr_epoch', '-lre', type=int, default=settings.RL_LR_EPOCH,
                        help='max epoch for reinforcement learning sampling epoch')
    parser.add_argument('--sample_step', '-ss', type=int, default=settings.RL_MAX_SAMPLE_STEP, 
                        help='the sample step of prune distribution')
    parser.add_argument('--sample_num', '-sn', type=int, default=settings.RL_MAX_SAMPLE_NUM, 
                        help='the sample number of prune distribution')
    parser.add_argument('--discount_factor', '-df', type=float, default=settings.RL_DISCOUNT_FACTOR, 
                        help='the discount factor for multi sample step')
    parser.add_argument('--step_length', '-sl', type=float, default=settings.RL_STEP_LENGTH, 
                        help='step length when updating prune distribution')
    parser.add_argument('--greedy_epsilon', '-ge', type=float, default=settings.RL_GREEDY_EPSILON, 
                        help='the probability to adopt random policy')
    parser.add_argument('--ppo', action='store_true', default=settings.RL_PPO_ENABLE, 
                        help='enable Proximal Policy Optimization')
    parser.add_argument('--ppo_clip', '-ppoc', type=float, default=settings.RL_PPO_CLIP, 
                        help='the clip value for PPO')
    parser.add_argument('--lr', '-lr', type=float, default=settings.T_FT_LR_SCHEDULER_INITIAL_LR,
                        help='initial fine tuning learning rate')
    parser.add_argument('--min_lr', '-mlr', type=float, default=settings.T_LR_SCHEDULER_MIN_LR,
                        help='minimal learning rate')
    parser.add_argument('--KD_temperature', '-KD_T', type=float, default=settings.T_FT_TEMPERATURE,
                        help='the tempearature used in knowledge distillation')
    parser.add_argument('--warmup_epoch', '-we', type=int, default=settings.T_WARMUP_EPOCH, 
                        help='warmup epoch number for lr scheduler')
    parser.add_argument('--fine_tune_epoch', '-fte', type=int, default=settings.T_FT_EPOCH,
                        help='fine tuning epoch for generated model')
    parser.add_argument('--stu_co', '-sc', type=float, default=settings.T_FT_STU_CO,
                        help='the student loss coefficient in knowledge distillation')
    parser.add_argument('--batch_size', '-b', type=int, default=settings.T_BATCH_SIZE, 
                        help='batch size for dataloader')
    parser.add_argument('--num_worker', '-n', type=int, default=settings.T_NUM_WORKER, 
                        help='number of workers for dataloader')
    parser.add_argument('--device', '-dev', type=str, default='cpu', 
                        help='device to use')
    parser.add_argument('--random_seed', '-rs', type=int, default=None, 
                        help='the random seed for the current new compression')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='resume the previous target compression')
    parser.add_argument('--resume_epoch', type=int, default=None, 
                        help='the specific previous compression epoch that to be resumed')
    parser.add_argument('--use_wandb', action='store_true', default=False, 
                        help='use wandb to track the experiment')
    
    parser.add_argument('--project_name', '-pn', type=str, default='RLPruner', 
                        help='the project name of wandb for the expriment')
    parser.add_argument('--log_dir', '-log', type=str, default='log', 
                        help='the directory containing logging text')
    parser.add_argument('--checkpoint_dir', '-ckptdir', type=str, default='checkpoint', 
                        help='the directory containing checkpoints')
    parser.add_argument('--pretrained_dir', '-ppth', type=str, default='pretrained_model', 
                        help='the directory containing pretrained model')
    parser.add_argument('--compressed_dir', '-cpth', type=str, default='compressed_model', 
                        help='the directory containing compressed model')

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args: argparse.Namespace):
    if args.resume == False:
        if args.model is None:
            raise ValueError(f"the specific model {args.model} should be provided")
        if args.dataset is None:
            raise ValueError(f"the specific type of dataset to train on should be provided, "
                             f"please select one of {DATASETS}")
        elif args.dataset not in DATASETS:
            raise ValueError(f"the provided dataset {args.dataset} is not supported, "
                             f"please select one of {DATASETS}")
    elif args.resume_epoch is None:
        raise ValueError(f"the specific resume-epoch {args.resume_epoch} should be provided, "
                         f"please specify which compression to resume")
    
    if args.prune_strategy not in PRUNE_STRATEGY:
        raise ValueError(f"the provided prune_strategy {args.prune_strategy} is not supported, "
                         f"please select one of {PRUNE_STRATEGY}")
    
    if args.sparsity >= 1 or args.sparsity < 0:
        raise ValueError(f"the sparsity of compressed model should be in interval [0, 1]")


if __name__ == '__main__':
    main()
