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


def extract_prunable_layers_info(model: nn.Module):
    prunable_layers = []
    output_dims = []

    def recursive_extract_prunable_layers_info(module, parent=None):
        children = list(module.children())
        for child in children:
            if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
                prunable_layers.append(child)
                if isinstance(child,(nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                    output_dims.append(child.out_channels)
                elif isinstance(child, nn.Linear):
                    output_dims.append(child.out_features)
            recursive_extract_prunable_layers_info(child, module)
    
    recursive_extract_prunable_layers_info(model)
    
    # remove the ouput layer as this layer out dim should equal to class num and can not be pruned
    # assumer the output layer is the last defined
    del prunable_layers[-1]
    del output_dims[-1]

    total_output_dim = sum(output_dims)
    filter_distribution = [dim / total_output_dim for dim in output_dims]
    
    return torch.tensor(filter_distribution), total_output_dim, prunable_layers


def extract_prunable_layer_dependence(model, x, prunable_layers):
    next_layers = [[] for _ in range(len(prunable_layers))]

    handles = []
    activation_hook_used = False
    all_layers = []
    get_tensor_recipients = {}
    get_layer_input_tensor = {}
    get_layer_output_tensor = {}

    def recursive_collect_all_layers(module, parent=None):
        children = list(module.children())
        for layer in children:
            if not list(layer.children()):
                all_layers.append(layer)
            recursive_collect_all_layers(layer, module)
    
    recursive_collect_all_layers(model)

    def forward_hook(layer, input, output):
        nonlocal activation_hook_used
        input = input[0]
        # ignore activation layer that operates tensor in_place and do not change tensor id
        if not list(layer.children()):
            if not (hasattr(layer, 'inplace')):
                activation_hook_used = False
                get_layer_input_tensor[layer] = input
                get_layer_output_tensor[layer] = output
                if id(input) not in get_tensor_recipients:
                    get_tensor_recipients[id(input)] = [layer]
                else:
                    get_tensor_recipients[id(input)].append(layer)

    for layer in all_layers:
        handle = layer.register_forward_hook(forward_hook)
        handles.append(handle)
    
    model.eval()
    model(x)

    for handle in handles:
        handle.remove()
    
    # check whether outputs during the inference are all unique respectively
    assert len(get_layer_output_tensor.values()) == len(set(get_layer_output_tensor.values()))
    
    # handle special layer
    for i, target_layer in enumerate(all_layers):
        if not (hasattr(target_layer, 'inplace') ):
            target_tensor = get_layer_output_tensor[target_layer]
            for layer, tensor in get_layer_input_tensor.items():
                if id(target_tensor) in get_tensor_recipients and layer in get_tensor_recipients[id(target_tensor)]:
                    continue
                if isinstance (layer, nn.Linear) and check_tensor_use_view(input_tensor=target_tensor, target_tensor=tensor):
                    # case 1: use x = x.view(x.size()[0], -1) before linear
                    if id(target_tensor) not in get_tensor_recipients:
                        print("1")
                        print(layer)
                        get_tensor_recipients[id(target_tensor)] = [layer]
                    else:
                        print("2")
                        print(layer)
                        get_tensor_recipients[id(target_tensor)].append(layer)
                    break
                elif check_tensor_in_concat(input_tensor=target_tensor, component_tensor=tensor, model=model):
                    # case 2: use torch.cat
                    if id(target_tensor) not in get_tensor_recipients:
                        print("3")
                        print(layer)
                        get_tensor_recipients[id(target_tensor)] = [layer]
                    else:
                        print("4")
                        print(layer)
                        get_tensor_recipients[id(target_tensor)].append(layer)
                    break
                elif target_tensor.shape == tensor.shape and check_tensor_residual(input_tensor=target_tensor, target_tensor=tensor, get_layer_output_tensor=get_layer_output_tensor):
                    # case 3: use residual short cut
                    if id(target_tensor) not in get_tensor_recipients:
                        print("5")
                        print(layer)
                        get_tensor_recipients[id(target_tensor)] = [layer]
                    else:
                        print("6")
                        print(layer)
                        get_tensor_recipients[id(target_tensor)].append(layer)
                    break
    
    for i, layer in enumerate(prunable_layers):
        relevant_tensors = queue.Queue()
        relevant_tensors.put(get_layer_output_tensor[layer])
        while not relevant_tensors.empty():
            cur_tensor = relevant_tensors.get()     # this will remove first tensor and get it
            cur_tensor_recipients = get_tensor_recipients[id(cur_tensor)]
            for recipient in cur_tensor_recipients:
                component_tensor = get_layer_input_tensor[recipient]
                next_layers[i].append([recipient, get_tensor_idx_at_next_layer(input_tensor=cur_tensor, component_tensor=component_tensor)])
                if not isinstance(recipient, (nn.Conv2d, nn.Linear)):
                    relevant_tensors.put(get_layer_output_tensor[recipient])
    
    # broadcast tensor idx offset at next linear layer
    for ith_next_layers in next_layers:
        pre_idx_offset = 0
        for next_layer_info in ith_next_layers:
            if next_layer_info[1] > 0:
                pre_idx_offset = next_layer_info[1]
            else:
                if isinstance(next_layer_info[0], nn.Linear):
                    next_layer_info[1] = pre_idx_offset

    return next_layers

def check_tensor_in_concat(input_tensor, component_tensor, model):
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
    flatten_input_tensor = input_tensor.view(input_tensor.size(0), -1)
    flatten_cat_size = flatten_input_tensor.size(dim)
    max_idx = component_tensor.size(dim) - cat_size + 1
    for i in range(max_idx):
        # case 1: conv -> conv, linear -> linear
        split = component_tensor[:, i:i+cat_size]
        if torch.equal(input_tensor, split):
            return i
    max_idx = component_tensor.size(dim) - flatten_cat_size + 1
    for i in range(max_idx):
        # case 2: conv -> linear
        split = component_tensor[:, i:i+flatten_cat_size]
        if torch.equal(flatten_input_tensor, split):
            return i
    return -1

def check_tensor_use_view(input_tensor, target_tensor):
    return torch.equal(input_tensor.view(input_tensor.size(0), -1), target_tensor.view(target_tensor.size(0), -1))

def check_tensor_residual(input_tensor, target_tensor, get_layer_output_tensor):
    return (target_tensor - input_tensor) in get_layer_output_tensor

def map_layers(original_net, generated_net):
    mapping = {}
    for orig_layer, gen_layer in zip(original_net.modules(), generated_net.modules()):
        if not list(orig_layer.children()):
            mapping[orig_layer] = gen_layer
    return mapping

def copy_prunable_and_next_layers(original_prunable_layuers, original_next_layers, mapping):
    generated_prunable_layers = []
    generated_next_layers = []
    for layer in original_prunable_layuers:
        if layer not in mapping:
            print(original_next_layers)
            print(original_prunable_layuers)
            print(mapping)
        generated_prunable_layers.append(mapping[layer])
    for layer_idx in range(len(original_next_layers)):
        new_layers_info = []
        for layer_info in original_next_layers[layer_idx]:
            new_layers_info.append([mapping[layer_info[0]], layer_info[1]])
        generated_next_layers.append(new_layers_info)
    return generated_prunable_layers, generated_next_layers
