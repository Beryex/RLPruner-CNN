from utils import extract_prunable_layers_info, extract_prunable_layer_dependence
import torch
device = 'cuda'

from models.googlenet import GoogleNet
from models.vgg import VGG16
ret = VGG16(3, 100).to(device)

_, _, layers = extract_prunable_layers_info(ret)
x = torch.rand(1, 3, 32, 32).to(device)
next_layers = extract_prunable_layer_dependence(ret, x, layers)
for i in range (len(layers)):
    print("Cur Conv/Linear:", layers[i])
    print("Next layers:", next_layers[i])

'''import torch
import torch.nn as nn

def check_tensor_in_concat(input_tensors, component_tensors, model):
    input_tensors.retain_grad()
    component_tensors.retain_grad()

    grad_output = torch.ones_like(component_tensors)
    
    model.zero_grad()
    component_tensors.backward(grad_output, retain_graph=True)
    
    start_idx = -1
    if input_tensors.grad is not None and torch.any(input_tensors.grad != 0).item():
        # split tensor and check one by one
        dim=1
        cat_size = input_tensors.size(dim)
        max_idx = component_tensors.size(dim) - cat_size + 1
        for i in range(max_idx):
            split = component_tensors[:, i:i+cat_size]
            if torch.equal(input_tensors, split):
                start_idx = i
                break
    if start_idx == -1:
        return False, -1
    else:
        return True, start_idx
    

class simple_model(nn.Module):
    def __init__(self, in_channels, num_class):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 2, kernel_size=5, bias=False)
        self.conv1 = nn.Conv2d(2, 5, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(2, 10, kernel_size=5, bias=False)
        self.conv3 = nn.Conv2d(15, num_class, kernel_size=5, bias=False)
    def forward(self, x):
        self.x0 = self.conv0(x)
        self.x1 = self.conv1(self.x0)
        self.x2 = self.conv2(self.x0)
        self.x3 = torch.cat([self.x1, self.x2], dim=1)
        return self.conv3(self.x3)

in_channels = 36
num_class = 10
model = simple_model(in_channels, num_class)
x = torch.randn(1, in_channels, 18, 18)

output = model(x)

result, index_start = check_tensor_in_concat(model.x2, model.x3, model)
print("x3 contains x2:", result)
print(index_start)'''
