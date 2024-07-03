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
    mask = []

    def recursive_extract_prunable_layers_info(module, parent=None):
        children = list(module.children())
        for child in children:
            if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
                prunable_layers.append(child)
                if isinstance(child, nn.Conv2d):
                    output_dims.append(child.out_channels)
                    mask.append(1)  # 1 for Conv2d
                elif isinstance(child, nn.Linear):
                    output_dims.append(child.out_features)
                    mask.append(0)  # 0 for Linear
            recursive_extract_prunable_layers_info(child, module)
    
    recursive_extract_prunable_layers_info(model)
    
    # remove the ouput layer as this layer out dim should equal to class num and can not be pruned
    last_fc_index = None
    for i, layer in enumerate(prunable_layers):
        if isinstance(layer, nn.Linear):
            last_fc_index = i

    if last_fc_index is not None:
        del prunable_layers[last_fc_index]
        del output_dims[last_fc_index]
        del mask[last_fc_index]

    total_output_dim = sum(output_dims)
    filter_distribution = [dim / total_output_dim for dim in output_dims]
    
    return torch.tensor(filter_distribution), torch.tensor(mask), prunable_layers


def extract_prunable_layer_dependence(model, x, prunable_layers):
    next_layers = [[] for _ in range(len(prunable_layers))]

    handles = []
    all_layers = []
    get_tensor_recipients = {}
    get_layer_input_tensor = {}
    get_layer_output_tensor = {}

    def recursive_collect_all_layers(module, parent=None):
        children = list(module.children())
        for child in children:
            nonlocal all_layers
            if not list(child.children()):
                all_layers.append(child)
            recursive_collect_all_layers(child, module)
    
    recursive_collect_all_layers(model)

    pre_output = None
    def forward_hook(layer, input, output):
        if not list(layer.children()):
            input = input[0]            # in hook function, input is passed in as tuple
            nonlocal pre_output
            nonlocal get_layer_input_tensor, get_layer_output_tensor, get_tensor_recipients
            get_layer_input_tensor[layer] = input
            get_layer_output_tensor[layer] = output
            if id(input) not in get_tensor_recipients:
                get_tensor_recipients[id(input)] = [layer]
            else:
                get_tensor_recipients[id(input)].append(layer)

    for layer in all_layers:
        handle = layer.register_forward_hook(forward_hook)
        handles.append(handle)
    
    with torch.no_grad():
        model(x)    

    for handle in handles:
        handle.remove()
    
    # find special layer that has residual input or torch.cat input
    for i, layer in enumerate(all_layers):
        if id(get_layer_input_tensor[layer]) not in get_tensor_recipients:
            component_tensor = get_layer_input_tensor[layer]
            for tensor in get_layer_output_tensor.values():
                if check_tensor_in_concat(input_tensors=tensor, component_tensors=component_tensor, model=model) == True:
                    get_tensor_recipients[id(tensor)].append(layer)
                    break
                
    for i, layer in enumerate(prunable_layers):
        print(next_layers)
        relevant_tensors = queue.Queue()
        relevant_tensors.put(get_layer_output_tensor[layer])
        while not relevant_tensors.empty():
            cur_tensor = relevant_tensors.get()     # this will remove first tensor and get it
            cur_tensor_recipients = get_tensor_recipients[id(cur_tensor)]
            for recipient in cur_tensor_recipients:
                print(recipient)
                component_tensor = get_layer_input_tensor[recipient]
                next_layers[i].append((recipient, get_tensor_idx_at_next_layer(input_tensors=cur_tensor, component_tensors=component_tensor)))
                if not isinstance(recipient, (nn.Conv2d, nn.Linear)):
                    relevant_tensors.put(get_layer_output_tensor[recipient])

    return next_layers

def check_tensor_in_concat(input_tensors, component_tensors, model):
    input_tensors.retain_grad()
    component_tensors.retain_grad()

    grad_output = torch.ones_like(component_tensors)
    
    model.zero_grad()
    component_tensors.backward(grad_output, retain_graph=True)
    
    start_idx = -1
    if input_tensors.grad is not None and torch.any(input_tensors.grad != 0).item():
        if get_tensor_idx_at_next_layer(input_tensors, component_tensors) == -1:
            return False
        else:
            return True
    else:
        return False

def get_tensor_idx_at_next_layer(input_tensors, component_tensors):
    # assmue always cat at dim=1
    dim=1
    cat_size = input_tensors.size(dim)
    max_idx = component_tensors.size(dim) - cat_size + 1
    for i in range(max_idx):
        split = component_tensors[:, i:i+cat_size]
        if torch.equal(input_tensors, split):
            return i
    return -1



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
