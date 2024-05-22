import torch
import time
import sys
import logging
import argparse
from thop import profile
from models.vgg import Custom_Conv2d, Custom_Linear
from tqdm import tqdm

from conf import settings
from utils import setup_logging, get_dataloader, count_custom_conv2d, count_custom_linear, torch_set_seed


def get_args():
    parser = argparse.ArgumentParser(description='train given model under given dataset')
    parser.add_argument('-net', type=str, default=None, help='the type of model to train')
    parser.add_argument('-dataset', type=str, default=None, help='the dataset to train on')
    parser.add_argument('-lr', type=float, default=settings.INITIAL_LR, help='initial learning rate')
    parser.add_argument('-lr_decay', type=float, default=settings.LR_DECAY, help='learning rate decay rate')
    parser.add_argument('-b', type=int, default=settings.BATCH_SIZE, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=settings.WARM, help='warm up training phase')

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

def test():
    original_para_num = 0.0
    original_FLOPs_num = 0.0
    compressed_para_num = 0.0
    compressed_FLOPs_num = 0.0

    correct_1 = 0.0
    correct_5 = 0.0

    net = torch.load('models/vgg16_cifar100_Original_1714258157.pkl').to(device)
    net.eval()
    with torch.no_grad():
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
    original_FLOPs_num, original_para_num = profile(net, inputs = (input, ), verbose=False, custom_ops = custom_ops)
    logging.info(f'Original model has top1 accuracy: {top1_acc}, top5 accuracy: {top5_acc}')
    logging.info(f'Original model has FLOPs: {original_FLOPs_num}, Parameter Num: {original_para_num}')
    
    correct_1 = 0.0
    correct_5 = 0.0

    compressed_net = torch.load('models/vgg16_cifar100_Compressed_1714336831.pkl').to(device)
    compressed_net.eval()
    with torch.no_grad():
        for (images, labels) in tqdm(test_loader, total=len(test_loader), desc='Testing round', unit='batch', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = compressed_net(images)
            _, preds = outputs.topk(5, 1, largest=True, sorted=True)
            correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
            top5_correct = labels.view(-1, 1).expand_as(preds) == preds
            correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(test_loader.dataset)
    top5_acc = correct_5 / len(test_loader.dataset)
    compressed_FLOPs_num, compressed_para_num = profile(compressed_net, inputs = (input, ), verbose=False, custom_ops = custom_ops)
    logging.info(f'Compressed model has top1 accuracy: {top1_acc}, top5 accuracy: {top5_acc}')
    logging.info(f'Compressed model has FLOPs: {compressed_FLOPs_num}, Parameter Num: {compressed_para_num}')

    FLOPs_compression_ratio = 1 - compressed_FLOPs_num / original_FLOPs_num
    Para_compression_ratio = 1 - compressed_para_num / original_para_num
    logging.info(f'FLOPS compressed ratio: {FLOPs_compression_ratio}, Parameter Num compressed ratio: {Para_compression_ratio}')
    logging.info(f'Original net: {net}')
    logging.info(f'Compressed net: {compressed_net}')
    

if __name__ == '__main__':
    start_time = int(time.time())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    setup_logging(experiment_id=start_time, net=args.net, dataset=args.dataset, action='test')

    torch_set_seed(start_time)
    torch.manual_seed(start_time)
    logging.info(f'Start with random seed: {start_time}')

    # process input argument
    _, _, test_loader, _, _ = get_dataloader(dataset=args.dataset, batch_size=args.b)
    if args.net == 'lenet5':
        input = torch.rand(1, 1, 32, 32).to(device)
    else:
        input = torch.rand(1, 3, 32, 32).to(device)
    custom_ops = {Custom_Conv2d: count_custom_conv2d, Custom_Linear: count_custom_linear}
    
    test()
    