"""
Copyright (c) 2024 Beryex, University of Illinois Urbana-Champaign
All rights reserved.

This code is licensed under the MIT License.
"""


import torch
import torch.nn as nn
import queue

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


def extract_layers_info(model: nn.Module):
    layers = []
    output_dims = []
    mask = []

    def recursive_extract_layers_info(module, parent=None):
        children = list(module.children())
        for child in children:
            if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
                layers.append(child)
                if isinstance(child, nn.Conv2d):
                    output_dims.append(child.out_channels)
                    mask.append(1)  # 1 for Conv2d
                elif isinstance(child, nn.Linear):
                    output_dims.append(child.out_features)
                    mask.append(0)  # 0 for Linear
            recursive_extract_layers_info(child, module)
    
    recursive_extract_layers_info(model)
    
    # remove the ouput layer as this layer out dim should equal to class num and can not be pruned
    last_fc_index = None
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            last_fc_index = i

    if last_fc_index is not None:
        del layers[last_fc_index]
        del output_dims[last_fc_index]
        del mask[last_fc_index]

    total_output_dim = sum(output_dims)
    filter_distribution = [dim / total_output_dim for dim in output_dims]
    
    return torch.tensor(filter_distribution), torch.tensor(mask), layers


def extract_layer_dependence(model, x, layers):
    next_layers = [[] for _ in range(len(layers))]

    handles = []
    all_layers = []
    get_tensor_recipients = {}
    get_layer_output_tensor = {}

    def recursive_collect_all_layers(module, parent=None):
        children = list(module.children())
        for child in children:
            nonlocal all_layers
            all_layers.append(child)
            recursive_collect_all_layers(child, module)
    
    recursive_collect_all_layers(model)

    def forward_hook(layer, input, output):
        input = input[0]
        output = output[0]
        if isinstance(input, tuple):
            print(layer)
            print(input)
            print(output)
            raise TypeError("Currently tuple input is not implemented")
        nonlocal get_tensor_recipients, get_layer_output_tensor
        get_layer_output_tensor[layer] = id(output)
        if isinstance(input.grad_fn, torch.autograd.function.CatBackward):
            # finish this part
            for cat_invoked_tensor in input.grad_fn.next_functions:
                if id(cat_invoked_tensor[0].variable) not in get_tensor_recipients:
                    get_tensor_recipients[id(cat_invoked_tensor[0].variable)] = [layer]
                else:
                    get_tensor_recipients[id(cat_invoked_tensor[0].variable)].append(layer)
        else:
            if id(input) not in get_tensor_recipients:
                get_tensor_recipients[id(input)] = [layer]
            else:
                get_tensor_recipients[id(input)].append(layer)

    for layer in all_layers:
        handle = layer.register_forward_hook(forward_hook)
        handles.append(handle)
    
    model(x)    # not use torch.no_grad to hook grad

    for handle in handles:
        handle.remove()

    for i, layer in enumerate(layers):
        relevant_tensors = queue.Queue()
        if layer not in get_layer_output_tensor:
            raise TypeError("There is layer defined but not used in forward process, check it")
        relevant_tensors.put(get_layer_output_tensor[layer])
        while not relevant_tensors.empty():
            cur_tensor_id = relevant_tensors.get()     # this will remove first tensor and get it
            cur_tensor_recipients = get_tensor_recipients[cur_tensor_id]
            for recipient in cur_tensor_recipients:
                next_layers[i].append(recipient)
                if not isinstance(recipient, (nn.Conv2d, nn.Linear)):
                    relevant_tensors.put(get_layer_output_tensor[recipient])

    return next_layers


'''found_conv_linear = False
        for next_layer in activated_layers[activated_layers.index(layer) + 1:]:
            if not found_conv_linear and isinstance(next_layer, (nn.Conv2d, nn.Linear)):
                next_conv_or_linear[i] = next_layer
                found_conv_linear = True
            if not found_norm and isinstance(next_layer, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
                next_norm[i] = next_layer
                found_norm = True
            if found_conv_linear and found_norm:
                break'''
