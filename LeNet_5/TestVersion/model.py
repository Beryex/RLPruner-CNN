from torch.nn import Module
from torch import nn
import torch


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.relu1 = torch.nn.ELU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(25, 23)
        self.relu3 = torch.nn.Tanh()
        self.fc2 = nn.Linear(23, 34)
        self.relu4 = torch.nn.ReLU()
        self.fc3 = nn.Linear(34, 10)
        self.relu5 = torch.nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y
