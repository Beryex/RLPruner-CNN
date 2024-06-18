import torch
import torch.nn as nn
import argparse
from thop import profile
from models.vgg import Custom_Conv2d, Custom_Linear
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_dataloader, count_custom_conv2d, count_custom_linear


def get_tensor_memory(tensor):
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.nelement()

def test_network(target_net: nn.Module):
    correct_1 = 0.0
    correct_5 = 0.0
    loss = 0.0
    target_net.eval()
    with torch.no_grad():
        for (images, labels) in tqdm(test_loader, total=len(test_loader), desc='Testing round', unit='batch', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = target_net(images)
            loss += loss_function(outputs, labels).item()
            _, preds = outputs.topk(5, 1, largest=True, sorted=True)
            correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
            top5_correct = labels.view(-1, 1).expand_as(preds) == preds
            correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(test_loader.dataset)
    top5_acc = correct_5 / len(test_loader.dataset)
    loss /= len(test_loader)
    FLOPs_num, para_num = profile(target_net, inputs = (input, ), verbose=False, custom_ops = custom_ops)
    return top1_acc, top5_acc, loss, FLOPs_num, para_num

def test_compression_result(original_net: nn.Module, 
                            compressed_net: nn.Module):
    top1_acc, top5_acc, loss, original_FLOPs_num, original_para_num = test_network(original_net)
    print(f'Original model has loss: {loss}, top1 accuracy: {top1_acc}, top5 accuracy: {top5_acc}')
    print(f'Original model has FLOPs: {original_FLOPs_num}, Parameter Num: {original_para_num}')

    top1_acc, top5_acc, loss, compressed_FLOPs_num, compressed_para_num = test_network(compressed_net)
    print(f'Compressed model has loss: {loss}, top1 accuracy: {top1_acc}, top5 accuracy: {top5_acc}')
    print(f'Compressed model has FLOPs: {compressed_FLOPs_num}, Parameter Num: {compressed_para_num}')

    FLOPs_compression_ratio = 1 - compressed_FLOPs_num / original_FLOPs_num
    Para_compression_ratio = 1 - compressed_para_num / original_para_num
    print(f'FLOPS compressed ratio: {FLOPs_compression_ratio}, Parameter Num compressed ratio: {Para_compression_ratio}')
    print(f'Original net: {original_net}')
    print(f'Compressed net: {compressed_net}')

def show_loss(train_loss: list, 
              test_loss: list, 
              top1_acc : list):
    epoch_list = list(range(len(train_loss)))
    plt.figure()

    plt.plot(epoch_list, train_loss, label='train_loss')
    plt.plot(epoch_list, test_loss, label='test_loss')
    plt.plot(epoch_list, top1_acc, label='top1_acc')

    plt.title('Train statics')
    plt.xlabel('Epoch')
    plt.ylabel('Statics')
    plt.legend()
    plt.show()
    

def get_args():
    parser = argparse.ArgumentParser(description='train given model under given dataset')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_function = nn.CrossEntropyLoss()
    args = get_args()

    # process input argument
    _, _, test_loader, _, _ = get_dataloader(dataset=args.dataset)
    original_net = torch.load('models/vgg16_cifar100_Original_1717669735.pth').to(device)
    compressed_net = torch.load('models/vgg16_cifar100_1717990275_finished.pth').to(device)
    
    if args.net == 'lenet5':
        input = torch.rand(1, 1, 32, 32).to(device)
    else:
        input = torch.rand(1, 3, 32, 32).to(device)
    custom_ops = {Custom_Conv2d: count_custom_conv2d, Custom_Linear: count_custom_linear}
    
    test_compression_result(original_net=original_net, compressed_net=compressed_net)