import os
import time
import argparse
import torch
import torch.nn as nn
from thop import profile
from typing import Dict
import logging
from prettytable import PrettyTable

from conf import settings
from utils import (get_dataloader, setup_logging, torch_set_random_seed, 
                   extract_prunable_layers_info, DATASETS)


def main():
    global args
    global device
    global eval_loader
    global dataset_name

    args = get_args()
    device = args.device
    model_name = args.model
    dataset_name = args.dataset

    """ Setup logging and get model, dataloader """
    random_seed = args.random_seed
    experiment_id = int(time.time())
    
    setup_logging(log_dir=args.log_dir,
                  experiment_id=experiment_id, 
                  random_seed=random_seed,
                  args=args,
                  model_name=args.model, 
                  dataset_name=args.dataset, 
                  action='evaluate',
                  use_wandb=False)
    
    torch_set_random_seed(random_seed)
    logging.info(f'Start with random seed: {random_seed}')
    print(f"Start with random seed: {random_seed}")
    
    pretrained_model = torch.load(f"{args.pretrained_pth}").to(device)
    compressed_model = torch.load(f"{args.compressed_pth}").to(device)
    _, _, test_loader, _ = get_dataloader(args.dataset, 
                                          batch_size=args.batch_size, 
                                          num_workers=args.num_worker)
    eval_loader = test_loader

    logging.info("Start evaluating")
    print(f"Start evaluating")
    results = {}
    results = evaluate_inference(pretrained_model, compressed_model, results)
    results = evaluate_model_stat(pretrained_model, compressed_model, results)
    results_table = PrettyTable()
    results_table.field_names = ["Metrics", 
                                 f"Pretrained {model_name}", 
                                 f"Compressed {model_name}", 
                                 "Improvement"]
    for metric, result in results.items():
        results_table.add_row([metric, result[0], result[1], result[2]], divider=True)
        logging.info(f"{metric}: "
                     f"Pretrained {model_name}: {result[0]} "
                     f"Comressed {model_name}: {result[1]} "
                     f"Improvement: {result[2]}")
    logging.info(f"Compressed model: {compressed_model}")
    print(results_table.get_string(title=f"Compression results of {model_name} on {dataset_name}"))


@torch.no_grad()
def evaluate_inference(pretrained_model: nn.Module,
                       compressed_model: nn.Module,
                       results: Dict) -> Dict:
    """ Evaluate pretrained_model and compressed_model on eval_loader """
    pretrained_model.eval()
    compressed_model.eval()

    correct_1_pre, correct_5_pre = 0.0, 0.0
    correct_1_com, correct_5_com = 0.0, 0.0

    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
          
        outputs_pre = pretrained_model(images)
        if isinstance(outputs_pre, tuple):
            outputs_pre = outputs_pre[0]
        _, preds_pre = outputs_pre.topk(5, 1, largest=True, sorted=True)
        correct_1_pre += (preds_pre[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct_pre = labels.view(-1, 1).expand_as(preds_pre) == preds_pre
        correct_5_pre += top5_correct_pre.any(dim=1).sum().item()

        outputs_com = compressed_model(images)
        if isinstance(outputs_com, tuple):
            outputs_com = outputs_com[0]
        _, preds_com = outputs_com.topk(5, 1, largest=True, sorted=True)
        correct_1_com += (preds_com[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct_com = labels.view(-1, 1).expand_as(preds_com) == preds_com
        correct_5_com += top5_correct_com.any(dim=1).sum().item()

    total_samples = len(eval_loader.dataset)
    top1_acc_pre = correct_1_pre / total_samples
    top5_acc_pre = correct_5_pre / total_samples
    top1_acc_com = correct_1_com / total_samples
    top5_acc_com = correct_5_com / total_samples
    results["top1_acc"] = (top1_acc_pre, 
                           top1_acc_com,
                           "{:.2f}%".format((top1_acc_com - top1_acc_pre) * 100))
    results["top5_acc"] = (top5_acc_pre, 
                           top5_acc_com,
                           "{:.2f}%".format((top5_acc_com - top5_acc_pre) * 100))

    return results


def evaluate_model_stat(pretrained_model: nn.Module,
                        compressed_model: nn.Module,
                        results: Dict) -> Dict:
    if dataset_name == 'mnist':
        sample_input = torch.rand(1, 1, 32, 32).to(device)
    else:
        sample_input = torch.rand(1, 3, 32, 32).to(device)

    FLOPs_pre, Para_pre = profile(model=pretrained_model, 
                                  inputs = (sample_input, ), 
                                  verbose=False)
    FLOPs_com, Para_com = profile(model=compressed_model, 
                                  inputs = (sample_input, ), 
                                  verbose=False)
    Mem_params_pre = sum([param.nelement()*param.element_size() for param in pretrained_model.parameters()])
    Mem_bufs_pre = sum([buf.nelement()*buf.element_size() for buf in pretrained_model.buffers()])
    Mem_pre = (Mem_params_pre + Mem_bufs_pre) / 1024 ** 3   # switch to GB
    Mem_params_com = sum([param.nelement()*param.element_size() for param in compressed_model.parameters()])
    Mem_bufs_com = sum([buf.nelement()*buf.element_size() for buf in compressed_model.buffers()])
    Mem_com = (Mem_params_com + Mem_bufs_com) / 1024 ** 3   # switch to GB
    file_size_pre = os.path.getsize(args.pretrained_pth) / 1024 ** 2   # switch to MB
    file_size_com = os.path.getsize(args.compressed_pth) / 1024 ** 2    # switch to MB
    _, Filter_Neuron_num_pre, _ = extract_prunable_layers_info(pretrained_model, [])
    _, Filter_Neuron_num_com, _ = extract_prunable_layers_info(compressed_model, [])
    
    results["FLOPs"] = (FLOPs_pre, 
                        FLOPs_com, 
                        "{:.2f}%".format((1 - FLOPs_com / FLOPs_pre) * 100))
    results["Para"] = (Para_pre, 
                       Para_com, 
                       "{:.2f}%".format((1 - Para_com / Para_pre) * 100))
    results["Filter and neuron number"] = (Filter_Neuron_num_pre,
                                           Filter_Neuron_num_com,
                                           "{:.2f}%".format((1 - Filter_Neuron_num_com / Filter_Neuron_num_pre) * 100))
    results["Mem (GB)"] = (Mem_pre, 
                           Mem_com, 
                           "{:.2f}%".format((1 - Mem_com / Mem_pre) * 100))
    results["File Size (MB)"] = (file_size_pre, 
                                 file_size_com, 
                                 "{:.2f}%".format((1 - file_size_com / file_size_pre) * 100))
    
    return results


def get_args():
    parser = argparse.ArgumentParser(description='Compress mode using RLPruner')
    parser.add_argument('--model', '-m', type=str, default=None, 
                        help='the name of model, just used to track logging')
    parser.add_argument('--dataset', '-ds', type=str, default=None, 
                        help='the dataset to train on')
    parser.add_argument('--batch_size', '-b', type=int, default=settings.T_BATCH_SIZE, 
                        help='batch size for dataloader')
    parser.add_argument('--num_worker', '-n', type=int, default=settings.T_NUM_WORKER, 
                        help='number of workers for dataloader')
    parser.add_argument('--device', '-dev', type=str, default='cpu', 
                        help='device to use')
    parser.add_argument('--random_seed', '-rs', type=int, default=1, 
                        help='the random seed for the evaluation')
    
    parser.add_argument('--log_dir', '-log', type=str, default='log', 
                        help='the directory containing logging text')
    parser.add_argument('--pretrained_pth', '-ppth', type=str, default='pretrained_model', 
                        help='the path of pretrained model')
    parser.add_argument('--compressed_pth', '-cpth', type=str, default='compressed_model', 
                        help='the path of compressed model')

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args: argparse.Namespace):
    if args.model is None:
        raise ValueError(f"the specific model {args.model} should be provided")
    if args.dataset is None:
        raise ValueError(f"the specific type of dataset to train on should be provided, "
                            f"please select one of {DATASETS}")
    elif args.dataset not in DATASETS:
        raise ValueError(f"the provided dataset {args.dataset} is not supported, "
                            f"please select one of {DATASETS}")


if __name__ == '__main__':
    main()
    