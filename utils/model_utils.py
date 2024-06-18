import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


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
            unpacked_tensor = self.weight_indices.unpack_tensors()
            actual_weight = self.weight[unpacked_tensor].view(self.weight_shape)
        actual_weight = actual_weight.to(x.device)
        return F.conv2d(x, actual_weight, self.bias, self.stride, self.padding)

    def prune_filter(self):
        if self.weight_indices is None:
            # weight is still normal, not cluster
            if self.out_channels - 1 == 0:
                return None
            weight_variances = torch.var(self.weight.data, dim = [1, 2, 3])
            # weight_L2norm = torch.norm(self.weight.data, p=2, dim= [1, 2, 3])
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
            unpacked_tensor = self.weight_indices.unpack_tensors()
            actual_weight = self.weight[unpacked_tensor].view(self.weight_shape)
        actual_weight = actual_weight.to(x.device)
        return F.linear(x, actual_weight, self.bias)
    

    def prune_filter(self):
        if self.weight_indices is None:
            # weight is still normal, not cluster
            if self.out_features - 1 == 0:
                return None
            weight_variances = torch.var(self.weight.data, dim = 1)
            # weight_L2norm = torch.norm(self.weight.data, p=2, dim=1)
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


class packable_tensors():
    def __init__(self, target_tensor: Tensor, num_clusters: int):
        self.target_tensor = target_tensor
        self.num_clusters = num_clusters
        self.original_dim = target_tensor.shape[0]

    def pack_tensors(self):
        input_tensor = self.target_tensor
        num_clusters = self.num_clusters
        if input_tensor.dtype != torch.int32 and input_tensor.dtype != torch.int64:
            raise TypeError(f"Unexpected input tensor dtype: {input_tensor.dtype} for packing tensor")
        # note that max value of num_clusters is 2048, which is limited by faiss k-means method
        if num_clusters <= 2 ** 11 and num_clusters > 2 ** 8:
            packed_tensors = input_tensor.to(torch.int16)
        elif num_clusters <= 2 ** 8 and num_clusters > 2 ** 4:
            input_tensor_original_dim = input_tensor.shape[0]
            packed_tensors_dim = input_tensor_original_dim // 2 + (1 if input_tensor_original_dim % 2 != 0 else 0)
            packed_tensors = torch.zeros(packed_tensors_dim, dtype=torch.int16)
            for i in range(0, input_tensor_original_dim - 1, 2):
                packed_tensors[i // 2] = ((input_tensor[i] & 0xFF) << 8) | (input_tensor[i+1] & 0xFF)
            if input_tensor_original_dim % 2 != 0:
                packed_tensors[-1] = (input_tensor[-1] & 0xFF) << 8
        elif num_clusters <= 2 ** 4:
            input_tensor_original_dim = input_tensor.shape[0]
            packed_tensors_dim = input_tensor_original_dim // 2 + (1 if input_tensor_original_dim % 2 != 0 else 0)
            packed_tensors = torch.zeros(packed_tensors_dim, dtype=torch.int8)
            for i in range(0, input_tensor_original_dim - 1, 2):
                packed_tensors[i // 2] = ((input_tensor[i] & 0x0F) << 4) | (input_tensor[i+1] & 0x0F)
            if input_tensor_original_dim % 2 != 0:
                packed_tensors[-1] = (input_tensor[-1] & 0x0F) << 4
        self.target_tensor = packed_tensors

    def unpack_tensors(self):
        input_tensor = self.target_tensor
        num_clusters = self.num_clusters
        original_dim = self.original_dim
        # note that max value of num_clusters is 2048, which is limited by faiss k-means method
        # target dtype is int32 as in forward the indexing only support int32 and does NOT support int16 or int8
        if num_clusters <= 2 ** 11 and num_clusters > 2 ** 8:
            unpacked_tensors = input_tensor.to(torch.int32)
        elif num_clusters <= 2 ** 8 and num_clusters > 2 ** 4:
            if input_tensor.dtype != torch.int16:
                raise TypeError(f"Unexpected input tensor dtype: {input_tensor.dtype} given {num_clusters} clusters number")
            unpacked_tensors = torch.zeros(original_dim, dtype=torch.int32)
            for i in range(0, input_tensor.shape[0] - 1):
                unpacked_tensors[2 * i] = (input_tensor[i] & 0xFF00) >> 8
                unpacked_tensors[2 * i + 1] = (input_tensor[i] & 0xFF)
            if original_dim % 2 == 0:
                unpacked_tensors[-1] = (input_tensor[-1] & 0xFF00) >> 8
            else:
                unpacked_tensors[-2] = (input_tensor[-1] & 0xFF00) >> 8
                unpacked_tensors[-1] = (input_tensor[-1] & 0xFF)
        elif num_clusters <= 2 ** 4:
            if input_tensor.dtype != torch.int8 and input_tensor.dtype != torch.uint8:
                raise TypeError(f"Unexpected input tensor dtype: {input_tensor.dtype} given {num_clusters} clusters number")
            unpacked_tensors = torch.zeros(original_dim, dtype=torch.int32)
            for i in range(0, input_tensor.shape[0] - 1):
                unpacked_tensors[2 * i] = (input_tensor[i] & 0xF0) >> 4
                unpacked_tensors[2 * i + 1] = (input_tensor[i] & 0x0F)
            if original_dim % 2 == 0:
                unpacked_tensors[-1] = (input_tensor[-1] & 0xF0) >> 4
            else:
                unpacked_tensors[-2] = (input_tensor[-1] & 0xF0) >> 4
                unpacked_tensors[-1] = (input_tensor[-1] & 0x0F)
        return unpacked_tensors


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
