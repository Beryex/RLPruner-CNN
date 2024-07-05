
import torch
from utils import extract_prunable_layer_dependence, extract_prunable_layers_info

device = 'cuda'
model = torch.load('./models/vgg16_cifar100_1720125382_original.pth').to(device)
from models.vgg import VGG16

# model = VGG16(3, 100).to(device)
sample_input = torch.rand(1, 3, 32, 32).to(device)
sample_input.requires_grad = True # used to extract dependence

prune_distribution, filter_num, prunable_layers = extract_prunable_layers_info(model)
print(prune_distribution)
next_layers = extract_prunable_layer_dependence(model=model, x=sample_input, prunable_layers=prunable_layers)
print(next_layers)

'''from utils import extract_prunable_layers_info, extract_prunable_layer_dependence
import torch
device = 'cuda'

from models.googlenet import GoogleNet
from models.vgg import VGG16
from models.unet import UNet
from models.lenet import LeNet5
from models.resnet import ResNet50
ret = GoogleNet(3, 100).to(device)

_, layers = extract_prunable_layers_info(ret)
x = torch.rand(1, 3, 32, 32).to(device)
x.requires_grad = True # used to extract dependence
next_layers = extract_prunable_layer_dependence(ret, x, layers)
for i in range (len(layers)):
    print("Cur Conv/Linear:", layers[i])
    print("Next layers:", next_layers[i])'''

'''import torch
import torch.nn as nn

def check_tensor_in_concat(input_tensor, component_tensor, model):
    print(input_tensor.is_leaf)
    print(component_tensor.is_leaf)
    input_tensor.retain_grad()
    component_tensor.retain_grad()

    grad_output = torch.ones_like(component_tensor)
    
    model.zero_grad()
    component_tensor.backward(grad_output, retain_graph=True)
    
    if input_tensor.grad is not None and torch.any(input_tensor.grad != 0).item():
        if get_tensor_idx_at_next_layer(input_tensor, component_tensor) == -1:
            return False
        else:
            return True
    else:
        return False

def get_tensor_idx_at_next_layer(input_tensor, component_tensor):
    # assmue always cat at dim=1
    dim=1
    cat_size = input_tensor.size(dim)
    max_idx = component_tensor.size(dim) - cat_size + 1
    for i in range(max_idx):
        split = component_tensor[:, i:i+cat_size]
        if torch.equal(input_tensor, split):
            print(i)
            return i
    return -1
    

class simple_model(nn.Module):
    def __init__(self, in_channels, num_class):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 2, kernel_size=5, bias=False)
        self.conv1 = nn.Conv2d(2, 5, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(2, 10, kernel_size=5, bias=False)
        self.conv3 = nn.Conv2d(15, num_class, kernel_size=5, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        self.x0 = self.conv0(x)
        self.x1 = self.conv1(self.x0)
        print(id(self.conv2(self.x0)))
        self.x2 = self.conv2(self.x0)
        print(id(self.x2))
        self.x4 = self.activation(self.x2)
        print(torch.equal(self.x2, self.x4))
        self.x3 = torch.cat([self.x1, self.x4], dim=1)
        return self.conv3(self.x3)

in_channels = 36
num_class = 10
model = simple_model(in_channels, num_class)
x = torch.full((1, in_channels, 18, 18), -1.0)

output = model(x)

result = check_tensor_in_concat(model.x2, model.x3, model)
print("x3 contains x2:", result)
'''
'''import torch
x1 = torch.rand(1, 3, 32, 32)
print(id(x1))
x2 = x1.view(x1.size()[0], -1)
print(id(x2))
print(torch.equal(x1.view(x1.size()[0], -1), x2.view(x2.size()[0], -1)))'''