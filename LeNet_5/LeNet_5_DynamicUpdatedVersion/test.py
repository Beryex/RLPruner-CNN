from model import LeNet
import numpy as np
import os
import torch
import visdom
import math
import copy
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def test():
    data_test = MNIST('./data/mnist',
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))
    # move the LeNet Module into the corresponding device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_loader = DataLoader(data_test, batch_size=1024, shuffle=True, num_workers=8)

    # initialize the testing parameters
    top1_correct_num = 0
    top3_correct_num = 0
    test_sample_num = 0

    # begin testing
    model = torch.load('models/mnist_0.pkl')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (test_x, test_label) in enumerate(test_loader):
            # move test data to device
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            # get predict y and predict its class
            predict_y = model(test_x.float()).detach()
            top1_correct_num += topk_correct_num(predict_y, test_label, 1)
            top3_correct_num += topk_correct_num(predict_y, test_label, 3)
            test_sample_num += test_label.size(0)
        # calculate the accuracy and print it
        top1_accuracy = top1_correct_num / test_sample_num
        top3_accuracy = top3_correct_num / test_sample_num
        print('model 0 has top1 accuracy: %f, top3 accuracy: %f' %(top1_accuracy, top3_accuracy))


def topk_correct_num(predict_y, test_label, topk = 1):
    _, pred = predict_y.topk(topk, dim=1, largest=True, sorted=True)
    correct = pred.eq(test_label.view(-1, 1).expand_as(pred))
    correct_k = correct.any(dim=1).float().sum().item()
    return correct_k

if __name__ == '__main__':
    test()