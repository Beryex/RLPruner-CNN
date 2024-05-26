import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from typing import Optional
from torch import Tensor

from conf import settings


class Custom_Conv2d(nn.Module):
    def __init__(self, 
                 in_channels: Optional[int] = None, 
                 out_channels: Optional[int] = None, 
                 kernel_size: Optional[torch.Size] = None, 
                 stride: int = 1, 
                 padding: int = 0, 
                 bias: bool = True, 
                 base_conv_layer: Optional[nn.Conv2d] = None):
        super(Custom_Conv2d, self).__init__()
        if base_conv_layer is None:
            if in_channels is None or out_channels is None or kernel_size is None:
                raise ValueError("Custom_Conv2d requires in_channels, out_channels, and kernel_size when no base_conv_layer is provided")
            temp_conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.weight = nn.Parameter(temp_conv.weight.data.detach())
            self.weight_shape = temp_conv.weight.shape
            if bias == True:
                self.bias = nn.Parameter(temp_conv.bias.data.detach())
            else:
                self.bias = None
            self.stride = stride
            self.padding = padding
        else:
            if not isinstance(base_conv_layer, nn.Conv2d):
                raise TypeError("base_conv_layer must be an instance of nn.Conv2d or its subclass")
            self.in_channels = base_conv_layer.in_channels
            self.out_channels = base_conv_layer.out_channels
            self.kernel_size = base_conv_layer.kernel_size
            self.weight = nn.Parameter(base_conv_layer.weight.data.detach())
            self.weight_shape = base_conv_layer.weight.shape
            if base_conv_layer.bias is not None:
                self.bias = nn.Parameter(base_conv_layer.bias.data.detach())
            else:
                self.bias = None
            self.stride = base_conv_layer.stride
            self.padding = base_conv_layer.padding
        
        # define parameter for weight sharing
        self.weight_indices = None
        self.original_weight = None
    
    def __repr__(self):
        return f'Custom_Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={False if self.bias is None else True})'
    
    def forward(self, 
                x: Tensor):
        if self.weight_indices is None:
            actual_weight = self.weight
        else:
            actual_weight = self.weight[self.weight_indices].view(self.weight_shape)
        actual_weight = actual_weight.to(x.device)
        return F.conv2d(x, actual_weight, self.bias, self.stride, self.padding)


    def prune_filter(self):
        if self.weight_indices is None:
            # weight is still normal, not cluster
            if self.out_channels - 1 == 0:
                return None
            weight_variances = torch.var(self.weight.data, dim = [1, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            with torch.no_grad():
                self.weight.data = torch.cat([self.weight.data[:target_kernel], self.weight.data[target_kernel+1:]], dim=0)
                if self.bias is not None:
                    self.bias.data = torch.cat([self.bias.data[:target_kernel], self.bias.data[target_kernel+1:]])
            self.out_channels -= 1
            self.weight_shape = self.weight.shape
            if self.out_channels != self.weight.shape[0]:
                raise ValueError(f'Conv2d layer out_channels {self.out_channels} and weight dimension {self.weight.shape[0]} mismath')
            return target_kernel
    
    def decre_input(self, 
                    target_kernel: int):
        with torch.no_grad():
            kept_indices = [i for i in range(self.in_channels) if i != target_kernel]
            self.weight.data = self.weight.data[:, kept_indices, :, :]
        self.in_channels -= 1
        self.weight_shape = self.weight.shape
        if self.in_channels != self.weight.shape[1]:
            raise ValueError(f'Conv2d layer in_channels {self.in_channels} and weight dimension {self.weight.shape[1]} mismath')
    
    def weight_sharing(self):
        if self.weight.shape[0] <= 2:
            return
        if self.original_weight is not None:
            original_weights = self.original_weight.data.clone().view(-1, 1).cuda()
            num_clusters = self.weight.shape[0] // settings.SCALING_FACTOR
        else:
            original_weights = self.weight.data.clone().view(-1, 1).cuda()
            num_clusters = min(len(original_weights) // settings.SCALING_FACTOR, 2048) # 2048 comes from faiss supports 2048 clusters at most

        kmeans = faiss.Kmeans(d=1, k=num_clusters, niter=50, verbose=False, gpu=True)
        kmeans.train(original_weights.cpu().numpy())

        clustered_weights = torch.from_numpy(kmeans.centroids).float().cuda().squeeze()
        labels = torch.from_numpy(kmeans.index.assign(original_weights.cpu().numpy(), k=1)).long().cuda()

        if self.weight_indices is None:
            self.original_weight = self.weight.detach()
        self.weight_indices = labels
        self.weight = torch.nn.Parameter(clustered_weights)
    
    def free_original_weight(self):
        if self.weight_indices is not None:
            self.original_weight = None

class Custom_Linear(nn.Module):
    def __init__(self, 
                 in_features: Optional[int] = None, 
                 out_features: Optional[int] = None, 
                 bias: bool = True, 
                 base_linear_layer: Optional[nn.Linear] = None):
        super(Custom_Linear, self).__init__()
        if base_linear_layer is None:
            if in_features is None or out_features is None:
                raise ValueError("Custom_Linear requires in_features, out_features when no base_linear_layer is provided")
            temp_linear = nn.Linear(in_features = in_features, out_features = out_features, bias = bias)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(temp_linear.weight.data.detach())
            self.weight_shape = temp_linear.weight.shape
            if bias == True:
                self.bias = nn.Parameter(temp_linear.bias.data.detach())
            else:
                self.bias = None
        else:
            if not isinstance(base_linear_layer, nn.Linear):
                raise TypeError("base_linear_layer must be an instance of nn.Linear or its subclass")
            self.weight = nn.Parameter(base_linear_layer.weight.data.detach())
            self.weight_shape = base_linear_layer.weight.shape
            if base_linear_layer.bias != None:
                self.bias = nn.Parameter(base_linear_layer.bias.data.detach())
            else:
                self.bias = None

        # define parameter for weight sharing
        self.weight_indices = None
        self.original_weight = None
    
    def __repr__(self):
        return f'Custom_Linear(in_features={self.in_features}, out_features={self.out_features}, bias={False if self.bias is None else True})'
    
    def forward(self, 
                x: Tensor):
        if self.weight_indices is None:
            actual_weight = self.weight
        else:
            actual_weight = self.weight[self.weight_indices].view(self.weight_shape)
        actual_weight = actual_weight.to(x.device)
        return F.linear(x, actual_weight, self.bias)
    

    def prune_filter(self):
        if self.weight_indices is None:
            # weight is still normal, not cluster
            if self.out_features - 1 == 0:
                return None
            weight_variances = torch.var(self.weight.data, dim = 1)
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                self.weight.data = torch.cat([self.weight.data[:target_neuron], self.weight.data[target_neuron+1:]], dim=0)
                if self.bias is not None:
                    self.bias.data = torch.cat([self.bias.data[:target_neuron], self.bias.data[target_neuron+1:]])
            self.out_features -= 1
            self.weight_shape = self.weight.shape
            if self.out_features != self.weight.shape[0]:
                raise ValueError(f'Linear layer out_channels {self.out_features} and weight dimension {self.weight.shape[0]} mismath')
            return target_neuron
    
    def decre_input(self, 
                    new_in_features: int, 
                    start_index: int, 
                    end_index: int):
        with torch.no_grad():
            self.weight.data = torch.cat([self.weight.data[:, :start_index], self.weight.data[:, end_index:]], dim=1)
            self.bias.data = self.bias.data
        self.in_features = new_in_features
        self.weight_shape = self.weight.shape
        if self.in_features != self.weight.shape[1]:
            raise ValueError(f'Linear layer in_channels {self.in_features} and weight dimension {self.weight.shape[1]} mismath')
    
    def weight_sharing(self):
        if self.weight.shape[0] <= 2:
            return
        if self.original_weight is not None:
            original_weights = self.original_weight.data.clone().view(-1, 1).cuda()
            num_clusters = self.weight.shape[0] // settings.SCALING_FACTOR
        else:
            original_weights = self.weight.data.clone().view(-1, 1).cuda()
            num_clusters = min(len(original_weights) // settings.SCALING_FACTOR, 2048) # 2048 comes from faiss supports 2048 clusters at most

        kmeans = faiss.Kmeans(d=1, k=num_clusters, niter=50, verbose=False, gpu=True)
        kmeans.train(original_weights.cpu().numpy())

        clustered_weights = torch.from_numpy(kmeans.centroids).float().cuda().squeeze()
        labels = torch.from_numpy(kmeans.index.assign(original_weights.cpu().numpy(), k=1)).long().cuda()

        if self.weight_indices is None:
            self.original_weight = self.weight.detach()
        self.weight_indices = labels
        self.weight = torch.nn.Parameter(clustered_weights)

    
    def free_original_weight(self):
        if self.weight_indices is not None:
            self.original_weight = None

def count_custom_conv2d(m: Custom_Conv2d, 
                        x: Tensor, 
                        y: Tensor):
    x = x[0]
    kernel_ops = m.weight_shape[2] * m.weight_shape[3]
    total_ops = y.nelement() * (m.weight_shape[1] * kernel_ops) # assume not using group convolution
    m.total_ops += torch.DoubleTensor([int(total_ops)])

def count_custom_linear(m: Custom_Linear, 
                        x: Tensor, 
                        y: Tensor):
    total_ops = y.nelement() * x[0].nelement() / y.size(0)
    m.total_ops += torch.DoubleTensor([int(total_ops)])

def get_network(net: str, 
                in_channels: int, 
                num_class: int):
    if net == 'vgg16':
        from models.vgg import VGG16
        ret = VGG16(in_channels, num_class)
    elif net == 'lenet5':
        from models.lenet import LeNet5
        ret = LeNet5(in_channels, num_class)
    elif net == 'googlenet':
        from models.googlenet import GoogleNet
        ret = GoogleNet(in_channels, num_class)
    elif net == 'resnet':
        from models.resnet import ResNet50
        ret = ResNet50(in_channels, num_class)
    elif net == 'unet':
        from models.unet import UNet
        ret = UNet(in_channels, num_class)
    else:
        ret = None

    return ret

def get_net_class(net: str):
    if net == 'vgg16':
        from models.vgg import VGG16
        net_class = VGG16
    elif net == 'lenet5':
        from models.lenet import LeNet5
        net_class = LeNet5
    elif net == 'googlenet':
        from models.googlenet import GoogleNet
        net_class = GoogleNet
    elif net == 'resnet':
        from models.resnet import ResNet50
        net_class = ResNet50
    elif net == 'unet':
        from models.unet import UNet
        net_class = UNet
    else:
        net_class = None

    return net_class
