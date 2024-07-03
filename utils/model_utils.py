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
        for layer in children:
            nonlocal all_layers
            # ignore activation layer as we assume it operates tensor in_place and do not change tensor id
            if not list(layer.children()) and not isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.PReLU)):
                all_layers.append(layer)
            recursive_collect_all_layers(layer, module)
    
    recursive_collect_all_layers(model)

    def forward_hook(layer, input, output):
        # ignore activation layer as we assume it operates tensor in_place and do not change tensor id
        if not list(layer.children()) and not isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.PReLU)):
            input = input[0]            # in hook function, input is passed in as tuple
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
    
    # check whether outputs during the inference are all unique respectively
    assert len(get_layer_output_tensor.values()) == len(set(get_layer_output_tensor.values()))
    
    # find special layer
    for i, target_layer in enumerate(all_layers):
        if id(get_layer_output_tensor[target_layer]) not in get_tensor_recipients:
            target_tensor = get_layer_output_tensor[target_layer]
            for layer, tensor in get_layer_input_tensor.items():
                if check_tensor_in_concat(input_tensor=target_tensor, component_tensor=tensor, model=model):
                    print("YES")
                    # case 1: use torch.cat
                    if id(target_tensor) not in get_tensor_recipients:
                        get_tensor_recipients[id(target_tensor)] = [layer]
                    else:
                        get_tensor_recipients[id(target_tensor)].append(layer)
                    break
                elif check_tensor_use_view(input_tensor=target_tensor, target_tensor=tensor):
                    # case 2: use x = x.view(x.size()[0], -1)
                    if id(target_tensor) not in get_tensor_recipients:
                        get_tensor_recipients[id(target_tensor)] = [layer]
                    else:
                        get_tensor_recipients[id(target_tensor)].append(layer)
                    break
                
    for i, layer in enumerate(prunable_layers):
        print(next_layers)
        relevant_tensors = queue.Queue()
        relevant_tensors.put(get_layer_output_tensor[layer])
        while not relevant_tensors.empty():
            cur_tensor = relevant_tensors.get()     # this will remove first tensor and get it
            cur_tensor_recipients = get_tensor_recipients[id(cur_tensor)]
            for recipient in cur_tensor_recipients:
                component_tensor = get_layer_input_tensor[recipient]
                next_layers[i].append((recipient, get_tensor_idx_at_next_layer(input_tensor=cur_tensor, component_tensor=component_tensor)))
                if not isinstance(recipient, (nn.Conv2d, nn.Linear)):
                    relevant_tensors.put(get_layer_output_tensor[recipient])

    return next_layers

def check_tensor_in_concat(input_tensor, component_tensor, model):
    input_tensor.requires_grad = True
    component_tensor.requires_grad = True
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
            return i
    return -1

def check_tensor_use_view(input_tensor, target_tensor):
    return torch.equal(input_tensor.view(input_tensor.size(0), -1), target_tensor.view(target_tensor.size(0), -1))


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
