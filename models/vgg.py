"""
Implementation of the VGG model, inspired by the architecture proposed in:
Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition."
Paper: https://arxiv.org/abs/1409.1556v6

This implementation is based on the PyTorch replication available at:
https://github.com/weiaicunzai/pytorch-cifar100/
"""


import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []

    input_channel = in_channels
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11(in_channels: int, num_class: int):
    return VGG(make_layers(cfg['A'], batch_norm=True, in_channels=in_channels), num_class)    

def vgg13(in_channels: int, num_class: int):
    return VGG(make_layers(cfg['B'], batch_norm=True, in_channels=in_channels), num_class)

def vgg16(in_channels: int, num_class: int):
    return VGG(make_layers(cfg['D'], batch_norm=True, in_channels=in_channels), num_class)

def vgg19(in_channels: int, num_class: int):
    return VGG(make_layers(cfg['E'], batch_norm=True, in_channels=in_channels), num_class)
