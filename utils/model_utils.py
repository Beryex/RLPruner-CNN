import torch
from torch import nn, Tensor
from typing import Tuple, List
import queue


# Define layer types for pruning and normalization
PRUNABLE_LAYERS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, 
                   nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
NORM_LAYERS = (nn.BatchNorm2d)
CONV_LAYERS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, 
               nn.ConvTranspose2d, nn.ConvTranspose3d)

# Define tensor comparision threshold for torch.allclose
# This is necessary because of tensor computation overflow
TENSOR_DIFFERENCE_THRESHOLD = 1e-4


def get_model(model_name: str, in_channels: int, num_class: int) -> nn.Module:
    """ Retrieve a specific network model based on the given specifications """
    if model_name == 'vgg16':
        from models.vgg import VGG16
        return VGG16(in_channels, num_class)
    elif model_name == 'lenet5':
        from models.lenet import LeNet5
        return LeNet5(in_channels, num_class)
    elif model_name == 'googlenet':
        from models.googlenet import GoogleNet
        return GoogleNet(in_channels, num_class)
    elif model_name == 'resnet50':
        from models.resnet import ResNet50
        return ResNet50(in_channels, num_class)
    elif model_name == 'unet':
        from models.unet import UNet
        return UNet(in_channels, num_class)
    else:
        return None


def extract_prunable_layers_info(model: nn.Module) -> Tuple[Tensor, int, List]:
    """ Extracts prunable layer information from a given neural network model """
    prunable_layers = []
    output_dims = []

    def recursive_extract_prunable_layers_info(module: nn.Module, parent: nn.Module):
        """ Recursively extracts prunable layers from a module """
        children = list(module.children())
        for child in children:
            if isinstance(child, PRUNABLE_LAYERS):
                prunable_layers.append(child)
                if isinstance(child, CONV_LAYERS):
                    output_dims.append(child.out_channels)
                elif isinstance(child, nn.Linear):
                    output_dims.append(child.out_features)
            recursive_extract_prunable_layers_info(child, module)
    
    recursive_extract_prunable_layers_info(model, None)
    
    # remove the ouput layer as its out dim should equal to class num and can not be pruned
    # assumer the output layer is the last defined
    del prunable_layers[-1]
    del output_dims[-1]

    total_output_dim = sum(output_dims)
    filter_distribution = [dim / total_output_dim for dim in output_dims]
    
    return torch.tensor(filter_distribution), total_output_dim, prunable_layers


def extract_prunable_layer_dependence(model: nn.Module, 
                                      x: Tensor,
                                      prunable_layers: List) -> Tuple[List, List]:
    """ Extract interdependence of prunable layers for end-to-end pruning """
    handles = []
    all_layers = []
    not_inplace_layers = []
    get_tensor_recipients = {}
    get_layer_input_tensor = {}
    get_layer_output_tensor = {}

    def recursive_collect_all_layers(module: nn.Module, parent: nn.Module) -> None:
        """ Recursively extracts prunable layers from a module """
        children = list(module.children())
        for layer in children:
            if not list(layer.children()):
                all_layers.append(layer)
            recursive_collect_all_layers(layer, module)
    
    recursive_collect_all_layers(model, None)

    def forward_hook(layer: nn.Module, input: Tuple, output: Tensor) -> None:
        """ Link each layer's I/O tensor to itself """
        input = input[0]
        # ignore activation layer that operates tensor in_place and not change tensor id
        if not list(layer.children()):
            if id(input) != id(output):
                not_inplace_layers.append(layer)
                get_layer_input_tensor[layer] = input
                # we add the random noise to make sure each layer has different output value
                get_layer_output_tensor[layer] = output.add_(torch.randn_like(output))
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
    for layer1, tensor1 in get_layer_output_tensor.items():
        for layer2, tensor2 in get_layer_output_tensor.items():
            if (id(tensor1) != id(tensor2) and 
                    tensor1.shape == tensor2.shape and 
                    torch.allclose(tensor1, tensor2, atol=TENSOR_DIFFERENCE_THRESHOLD)):
                raise ValueError(f'{layer1} and {layer2} has the same value output')
    
    """ handle special layer """
    for output_layer, output_tensor in get_layer_output_tensor.items():
        for input_layer, input_tensor in get_layer_input_tensor.items():
            if (id(output_tensor) in get_tensor_recipients 
                    and input_layer in get_tensor_recipients[id(output_tensor)]):
                continue
            if (isinstance (input_layer, nn.Linear) 
                    and check_tensor_use_view(output_tensor, input_tensor)):
                # case 1: use x = x.view(x.size()[0], -1) before linear
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)
            
            elif check_tensor_in_concat(output_tensor, input_tensor):
                # case 2: use torch.cat
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)
            
            elif check_tensor_addition(output_tensor, input_tensor, get_layer_output_tensor):
                # case 3: input_tensor = output_tensor + another output_tensor
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)
            
            elif check_tensor_residual(output_tensor, input_tensor, get_layer_input_tensor):
                # case 4: use residual short cut: input_tensor = output_tensor + another input_tensor
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)

    """ linke each layer's next layers """
    next_layers = [[] for _ in range(len(prunable_layers))]
    for i, layer in enumerate(prunable_layers):
        relevant_tensors = queue.Queue()
        relevant_tensors.put(get_layer_output_tensor[layer])
        while not relevant_tensors.empty():
            cur_tensor = relevant_tensors.get()
            cur_tensor_recipients = get_tensor_recipients[id(cur_tensor)]
            for recipient in cur_tensor_recipients:
                component_tensor = get_layer_input_tensor[recipient]
                element = [recipient, get_tensor_idx_at_next_layer(cur_tensor, component_tensor)]
                next_layers[i].append(element)
                if not isinstance(recipient, PRUNABLE_LAYERS):
                    relevant_tensors.put(get_layer_output_tensor[recipient])
    
    """ build mask for each prunable layer to indicate the residual layer """
    # cluster 0 means this layer is independent and can be pruned directly
    # any value greater than 1 means this layer is in a residual cluster with 
    # the same that value and all layers inside a group should be pruned together
    layer_cluster_mask = [0 for _ in range(len(prunable_layers))]
    cur_cluster_mask = 1
    involved_tensors = []
    for layer_idx, layer in enumerate(prunable_layers):
        for next_layer_info in next_layers[layer_idx]:
            next_layer = next_layer_info[0]
            offset = next_layer_info[1]
            if isinstance(next_layer, NORM_LAYERS):
                layer = next_layer
            elif isinstance(next_layer, (CONV_LAYERS, nn.AdaptiveAvgPool2d)):
                if offset == -1:
                    # if the connection offset mismatch, there exists residual layer
                    # check if it is previous residual cluster
                    if len(involved_tensors) == 0:
                        involved_tensors.append(get_layer_output_tensor[layer])
                        involved_tensors.append(get_layer_input_tensor[next_layer])
                        layer_cluster_mask[layer_idx] = cur_cluster_mask
                    else:
                        output_tensor1 = get_layer_output_tensor[layer]
                        component_tensor = get_layer_input_tensor[next_layer]
                        target_tensor = component_tensor - output_tensor1
                        in_previous_cluster = False
                        for output_tensor2 in involved_tensors:
                            if (target_tensor.shape == output_tensor2.shape and
                                    torch.allclose(target_tensor, output_tensor2, 
                                                   atol=TENSOR_DIFFERENCE_THRESHOLD)):
                                # indicates the new layer is still in prevuous residual cluster
                                involved_tensors.append(get_layer_output_tensor[layer])
                                involved_tensors.append(get_layer_input_tensor[next_layer])
                                layer_cluster_mask[layer_idx] = cur_cluster_mask
                                in_previous_cluster = True
                                break
                        if in_previous_cluster == False:
                            # indicates the new layer is starting a new cluster
                            cur_cluster_mask += 1
                            involved_tensors = []
                            involved_tensors.append(get_layer_output_tensor[layer])
                            involved_tensors.append(get_layer_input_tensor[next_layer])
                            layer_cluster_mask[layer_idx] = cur_cluster_mask
    
    """ broadcast tensor idx offset at next linear layer """
    for ith_next_layers in next_layers:
        pre_idx_offset = 0
        for next_layer_info in ith_next_layers:
            if next_layer_info[1] > 0:
                pre_idx_offset = next_layer_info[1]
            else:
                if isinstance(next_layer_info[0], nn.Linear):
                    next_layer_info[1] = pre_idx_offset

    return next_layers, layer_cluster_mask

def check_tensor_in_concat(input_tensor: Tensor, component_tensor: Tensor) -> bool:
    """ Check whether component_tensor is acquired by concatenating using input_tensor """
    if input_tensor.shape[2:] != component_tensor.shape[2:]:
        return False
    
    if get_tensor_idx_at_next_layer(input_tensor, component_tensor) == -1:
        return False
    else:
        return True

def get_tensor_idx_at_next_layer(input_tensor: Tensor, component_tensor: Tensor) -> int:
    """ get the starting index of input_tensor in component_tensor, -1 if fail """
    dim=1   # assmue always cat at dim=1
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

def check_tensor_use_view(input_tensor: Tensor, target_tensor: Tensor) -> bool:
    """ Check whether target_tensor is acquired using input_tensor.view() """
    return torch.equal(input_tensor.view(input_tensor.size(0), -1), 
                       target_tensor.view(target_tensor.size(0), -1))

def check_tensor_addition(input_tensor: Tensor, 
                          component_tensor: Tensor, 
                          get_layer_output_tensor: dict) -> bool:
    """ Check whether component_tensor == input_tensor + another layer's output """
    if component_tensor.shape != input_tensor.shape:
        return False
    
    residual_tensor = component_tensor - input_tensor
    for tensor in get_layer_output_tensor.values():
        if tensor.shape == residual_tensor.shape:
            if torch.allclose(tensor, residual_tensor, atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
    return False

def check_tensor_residual(input_tensor: Tensor, 
                          component_tensor: Tensor, 
                          get_layer_input_tensor: dict) -> bool:
    """ Check whether component tensor == input_tensor + another layer's input """
    if component_tensor.shape != input_tensor.shape:
        return False
    
    residual_tensor = component_tensor - input_tensor
    for tensor in get_layer_input_tensor.values():
        if tensor.shape == residual_tensor.shape:
            if torch.allclose(tensor, residual_tensor, atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
    return False

def adjust_prune_distribution_for_cluster(prune_distribution: Tensor, 
                                          layer_cluster_mask: List) -> Tensor:
    """ Adjust so that layer among non-zero cluster has equal probability to be pruned """
    cluster_total_value = {}
    cluster_layer_number = {}
    for idx, mask in enumerate(layer_cluster_mask):
        if mask > 0:
            if mask not in cluster_total_value:
                cluster_total_value[mask] = prune_distribution[idx].item()
                cluster_layer_number[mask] = 1
            else:
                cluster_total_value[mask] += prune_distribution[idx].item()
                cluster_layer_number[mask] += 1
    for idx, mask in enumerate(layer_cluster_mask):
        if mask > 0:
            prune_distribution[idx] = cluster_total_value[mask] / cluster_layer_number[mask]
    return prune_distribution
