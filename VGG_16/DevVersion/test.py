import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader

from thop import profile

import sys


# move the LeNet Module into the corresponding device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test():
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=1024,
        shuffle=True
    )

    original_para_num = 0.0
    original_FLOPs_num = 0.0
    compressed_para_num = 0.0
    compressed_FLOPs_num = 0.0
    input = torch.rand(128, 3, 32, 32).to(device)

    # initialize the testing parameters
    top1_correct_num = 0.0
    top5_correct_num = 0.0

    # begin testing
    model = torch.load('models/VGG_Original_1709433114.pkl')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (test_x, test_label) in enumerate(cifar100_test_loader):
            # move test data to device
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            # get predict y and predict its class
            outputs = model(test_x)
            _, preds = outputs.topk(5, 1, largest=True, sorted=True)
            top1_correct_num += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
            top5_correct = test_label.view(-1, 1).expand_as(preds) == preds
            top5_correct_num += top5_correct.any(dim=1).sum().item()
        # calculate the accuracy and print it
        top1_accuracy = top1_correct_num / len(cifar100_test_loader.dataset)
        top5_accuracy = top5_correct_num / len(cifar100_test_loader.dataset)
        print('Original model has top1 accuracy: %f, top5 accuracy: %f' %(top1_accuracy, top5_accuracy))
        original_FLOPs_num, original_para_num = profile(model, inputs = (input, ), verbose=False)
        
    
    # initialize the testing parameters
    top1_correct_num = 0.0
    top5_correct_num = 0.0

    # begin testing
    model = torch.load('models/VGG_Compressed_1708879256.pkl')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (test_x, test_label) in enumerate(cifar100_test_loader):
            # move test data to device
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            # get predict y and predict its class
            outputs = model(test_x)
            _, preds = outputs.topk(3, 1, largest=True, sorted=True)
            top1_correct_num += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
            top5_correct = test_label.view(-1, 1).expand_as(preds) == preds
            top5_correct_num += top5_correct.any(dim=1).sum().item()
        # calculate the accuracy and print it
        top1_accuracy = top1_correct_num / len(cifar100_test_loader.dataset)
        top5_accuracy = top5_correct_num / len(cifar100_test_loader.dataset)
        '''print(model.features)
        print(model.classifier)
        print(model.features['Conv6'].weight.shape)
        print(model.classifier[3].weight.shape)'''
        print('Compressed Model has top1 accuracy: %f, top5 accuracy: %f' %(top1_accuracy, top5_accuracy))
        compressed_FLOPs_num, compressed_para_num = profile(model, inputs = (input, ), verbose=False)
    
    # get compressed ratio
    FLOPs_compressed_ratio = compressed_FLOPs_num / original_FLOPs_num
    Para_compressed_ratio = compressed_para_num / original_para_num
    print('We achieve FLOPS compressed ratio: %f, parameter number compressed ratio: %f' %(FLOPs_compressed_ratio, Para_compressed_ratio))

if __name__ == '__main__':
    test()