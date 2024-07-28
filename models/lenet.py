import torch
from torch import Tensor
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, in_channels: int=1, num_class: int=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(84, num_class)
        )
        
    
    def forward(self, x: Tensor):
        x = self.conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_layers(x)
        return x
