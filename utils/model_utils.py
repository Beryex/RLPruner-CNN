import torch
from torch import nn, Tensor
from typing import Tuple, List, Dict
import queue


# Define layer types for pruning and normalization
PRUNABLE_LAYERS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, 
                   nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
NORM_LAYERS = (nn.BatchNorm2d, nn.BatchNorm1d)
CONV_LAYERS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, 
               nn.ConvTranspose2d, nn.ConvTranspose3d)

# Define supported models
MODELS = ["vgg11", "vgg13", "vgg16", "vgg19",
          "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
          "resnet8", "resnet14", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110",
          "densenet121", "densenet161", "densenet169", "densenet201",
          "mobilenetv3_small", "mobilenetv3_large",
          "googlenet"]

# Define tensor comparision threshold for torch.allclose
# This is necessary because of tensor computation overflow
TENSOR_DIFFERENCE_THRESHOLD = 1e-2  # this seems too large, but necessary for some case overflow
COMPARE_LENGTH = 5


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """ Retrieve a specific network model based on the given specifications """
    in_channels = 3
    if model_name == 'vgg11':
        from models import vgg11
        return vgg11(in_channels, num_classes)
    elif model_name == 'vgg13':
        from models import vgg13
        return vgg13(in_channels, num_classes)
    elif model_name == 'vgg16':
        from models import vgg16
        return vgg16(in_channels, num_classes)
    elif model_name == 'vgg19':
        from models import vgg19
        return vgg19(in_channels, num_classes)
    elif model_name == 'resnet18':
        from models import resnet18
        return resnet18(in_channels, num_classes)
    elif model_name == 'resnet34':
        from models import resnet34
        return resnet34(in_channels, num_classes)
    elif model_name == 'resnet50':
        from models import resnet50
        return resnet50(in_channels, num_classes)
    elif model_name == 'resnet101':
        from models import resnet101
        return resnet101(in_channels, num_classes)
    elif model_name == 'resnet152':
        from models import resnet152
        return resnet152(in_channels, num_classes)
    elif model_name == 'resnet8':
        from models import resnet8
        return resnet8(in_channels, num_classes)
    elif model_name == 'resnet14':
        from models import resnet14
        return resnet14(in_channels, num_classes)
    elif model_name == 'resnet20':
        from models import resnet20
        return resnet20(in_channels, num_classes)
    elif model_name == 'resnet32':
        from models import resnet32
        return resnet32(in_channels, num_classes)
    elif model_name == 'resnet44':
        from models import resnet44
        return resnet44(in_channels, num_classes)
    elif model_name == 'resnet56':
        from models import resnet56
        return resnet56(in_channels, num_classes)
    elif model_name == 'resnet110':
        from models import resnet110
        return resnet110(in_channels, num_classes)
    elif model_name == 'densenet121':
        from models import densenet121
        return densenet121(in_channels, num_classes)
    elif model_name == 'densenet209':
        from models import densenet169
        return densenet169(in_channels, num_classes)
    elif model_name == 'densenet201':
        from models import densenet201
        return densenet201(in_channels, num_classes)
    elif model_name == 'densenet161':
        from models import densenet161
        return densenet161(in_channels, num_classes)
    elif model_name == 'mobilenetv3_small':
        from models import mobilenetv3_small
        return mobilenetv3_small(in_channels, num_classes)
    elif model_name == 'mobilenetv3_large':
        from models import mobilenetv3_large
        return mobilenetv3_large(in_channels, num_classes)
    elif model_name == 'googlenet':
        from models import googlenet
        return googlenet(in_channels, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def set_inplace_false(model: nn.Module, original_inplace_state: Dict):
    """ Turn off all the inplace execution for extracting layer interdependence later """
    for name, module in model.named_modules():
        if hasattr(module, 'inplace'):
            original_inplace_state[name] = module.inplace
            module.inplace = False


def recover_inplace_status(model: nn.Module, original_inplace_state: Dict):
    """ Recover the inplace status of layer in the model """
    for name, module in model.named_modules():
        if name in original_inplace_state:
            module.inplace = original_inplace_state[name]


@torch.no_grad()
def extract_prunable_layers_info(model: nn.Module, skip_layer_index: List) -> Tuple[Tensor, int, List]:
    """ Extracts prunable layer information from a given neural network model """
    prunable_layers = []
    output_dims = []

    def recursive_extract_prunable_layers_info(module: nn.Module):
        """ Recursively extracts prunable layers from a module """
        children = list(module.children())
        for child in children:
            if isinstance(child, PRUNABLE_LAYERS):
                prunable_layers.append(child)
                if isinstance(child, CONV_LAYERS):
                    output_dims.append(child.out_channels)
                elif isinstance(child, nn.Linear):
                    output_dims.append(child.out_features)
            recursive_extract_prunable_layers_info(child)
    
    recursive_extract_prunable_layers_info(model)
    
    # skip the ouput layer as its out dim should equal to class num and can not be pruned
    del prunable_layers[-1]
    del output_dims[-1]

    prunable_layers = [item for idx, item in enumerate(prunable_layers) if idx not in skip_layer_index]
    output_dims = [item for idx, item in enumerate(output_dims) if idx not in skip_layer_index]

    total_output_dim = sum(output_dims)
    filter_distribution = [dim / total_output_dim for dim in output_dims]
    
    return torch.tensor(filter_distribution), total_output_dim, prunable_layers


@torch.no_grad()
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

    def recursive_collect_all_layers(module: nn.Module) -> None:
        """ Recursively extracts prunable layers from a module """
        children = list(module.children())
        for layer in children:
            if not list(layer.children()):
                all_layers.append(layer)
            recursive_collect_all_layers(layer)
    
    recursive_collect_all_layers(model)

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
    
    """ link special layer """
    for output_layer, output_tensor in get_layer_output_tensor.items():
        for input_layer, input_tensor in get_layer_input_tensor.items():
            if (id(input_layer) == id(output_layer)):
                continue
            if (isinstance (input_layer, nn.Linear) 
                    and check_tensor_use_view(output_tensor, input_tensor)):
                # case 1: use x = x.view(x.size()[0], -1) or flatten(1) before linear
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)
            
            elif check_tensor_addition(output_tensor, input_tensor, get_layer_output_tensor):
                # case 2: input_tensor = output_tensor + another output_tensor
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)
            
            elif check_tensor_residual(output_tensor, input_tensor, get_layer_input_tensor):
                # case 3: use residual short cut: input_tensor = output_tensor + another input_tensor
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)
            
            elif check_tensor_in_concat(output_tensor, input_tensor):
                # case 4: use torch.cat
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)
            
            elif check_tensor_squeeze_excitation(output_tensor, input_tensor, get_layer_output_tensor):
                # case 5: use squeeze excitation: input_tensor = output_tensor * another output
                if id(output_tensor) not in get_tensor_recipients:
                    get_tensor_recipients[id(output_tensor)] = [input_layer]
                elif input_layer not in get_tensor_recipients[id(output_tensor)]:
                    get_tensor_recipients[id(output_tensor)].append(input_layer)

    '''""" Check layer Linkage (combined with print model for debug purpose) """
    for layer, tensor in get_layer_output_tensor.items():
        print("\n")
        print(layer)
        if id(tensor) in get_tensor_recipients:
            print(get_tensor_recipients[id(tensor)])'''

    """ linke each layer's next layers using queue """
    next_layers = [[] for _ in range(len(prunable_layers))]
    for i, layer in enumerate(prunable_layers):
        relevant_tensors = queue.Queue()
        relevant_tensors.put(get_layer_output_tensor[layer])
        while not relevant_tensors.empty():
            cur_tensor = relevant_tensors.get()
            cur_tensor_recipients = get_tensor_recipients[id(cur_tensor)]
            for recipient in cur_tensor_recipients:
                component_tensor = get_layer_input_tensor[recipient]
                # skip inplace execution layer
                element = [recipient, get_tensor_idx_at_next_layer(cur_tensor, component_tensor)]
                next_layers[i].append(element)
                # stop at the first prunable layers at current branch
                if not isinstance(recipient, PRUNABLE_LAYERS):
                    relevant_tensors.put(get_layer_output_tensor[recipient])
    
    """ build mask for each prunable layer to indicate the residual layer """
    # cluster 0 means this layer is independent and can be pruned directly
    # any value greater than 1 means this layer is in a residual cluster with 
    # the same that value and all layers inside a group should be pruned together
    layer_cluster_mask = [0 for _ in range(len(prunable_layers))]
    cur_cluster_max_mask = 1
    cur_cluster_tensors = {}
    for layer_idx, layer in enumerate(prunable_layers):
        for next_layer_info in next_layers[layer_idx]:
            next_layer = next_layer_info[0]
            offset = next_layer_info[1]
            if offset >= 0:
                layer = next_layer
            elif offset == -1:
                # if the connection offset mismatch, there exists residual layer
                if cur_cluster_max_mask not in cur_cluster_tensors:
                    # if no previous residual cluster, create one
                    cur_cluster_tensors[cur_cluster_max_mask] = []
                    cur_cluster_tensors[cur_cluster_max_mask].append(get_layer_input_tensor[layer])
                    cur_cluster_tensors[cur_cluster_max_mask].append(get_layer_output_tensor[layer])
                    cur_cluster_tensors[cur_cluster_max_mask].append(get_layer_input_tensor[next_layer])
                    cur_cluster_tensors[cur_cluster_max_mask].append(get_layer_output_tensor[next_layer])
                    layer_cluster_mask[layer_idx] = cur_cluster_max_mask
                else:
                    # else, we need to check if it belongs to the previous residual cluster
                    output_tensor = get_layer_output_tensor[layer]
                    component_tensor = get_layer_input_tensor[next_layer]
                    find_existing = False
                    for cur_cluster_mask in range(1, 1 + cur_cluster_max_mask):
                        if check_tensor_in_cluster(output_tensor, 
                                                component_tensor, 
                                                cur_cluster_tensors[cur_cluster_mask]):
                            # indicates the new layer is still in prevuous residual cluster
                            cur_cluster_tensors[cur_cluster_mask].append(get_layer_input_tensor[layer])
                            cur_cluster_tensors[cur_cluster_mask].append(get_layer_output_tensor[layer])
                            cur_cluster_tensors[cur_cluster_mask].append(get_layer_input_tensor[next_layer])
                            cur_cluster_tensors[cur_cluster_mask].append(get_layer_output_tensor[next_layer])
                            layer_cluster_mask[layer_idx] = cur_cluster_mask
                            find_existing = True
                    if find_existing == False:
                        # indicates the new layer is starting a new cluster
                        cur_cluster_max_mask += 1
                        cur_cluster_tensors[cur_cluster_max_mask] = []
                        cur_cluster_tensors[cur_cluster_max_mask].append(get_layer_input_tensor[layer])
                        cur_cluster_tensors[cur_cluster_max_mask].append(get_layer_output_tensor[layer])
                        cur_cluster_tensors[cur_cluster_max_mask].append(get_layer_input_tensor[next_layer])
                        cur_cluster_tensors[cur_cluster_max_mask].append(get_layer_output_tensor[next_layer])
                        layer_cluster_mask[layer_idx] = cur_cluster_max_mask
                break
    
    """ Move mask for each grouped convolution layer to its previous conv layer as we skip them during pruning """
    # Case: (conv -> grouped conv) + (conv -> grouped conv) -> conv
    # TO DO: this part does not work when (grouped conv -> grouped conv) + (grouped conv -> grouped conv) -> conv
    for idx, layer in enumerate(prunable_layers):
        if isinstance(layer, nn.Conv2d) and layer.groups > 1 and layer_cluster_mask[idx] > 0:
            target_layer = layer
            target_mask = layer_cluster_mask[idx]
            layer_cluster_mask[idx] = 0
            for target_idx, next_layers_info in enumerate(next_layers):
                for next_layer_info in next_layers_info:
                    next_layer = next_layer_info[0]
                    if id(next_layer) == id(target_layer):
                        layer_cluster_mask[target_idx] = target_mask
    
    """ broadcast tensor idx offset at next prunable layer """
    # this is because the use of maxpool2d or x.view(x.size()[0], -1)
    # that integrate inputs from different layer into a single output
    # so we have to broadcast offset manually.
    for ith_next_layers in next_layers:
        pre_idx_offset = 0
        for next_layer_info in ith_next_layers:
            if next_layer_info[1] > 0:
                pre_idx_offset = next_layer_info[1]
            elif next_layer_info[1] == 0:
                if isinstance(next_layer_info[0], PRUNABLE_LAYERS):
                    next_layer_info[1] = pre_idx_offset
            else:
                # this occur when offset is -1, indicating residual shortcut
                continue

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
    input_view = input_tensor.view(input_tensor.size(0), -1)
    target_view = target_tensor.view(target_tensor.size(0), -1)
    
    if input_view.size(1) > COMPARE_LENGTH:
        return torch.equal(input_view[:, :COMPARE_LENGTH], target_view[:, :COMPARE_LENGTH])
    
    return torch.equal(input_view, target_view)


def check_tensor_addition(input_tensor: Tensor, 
                          component_tensor: Tensor, 
                          get_layer_output_tensor: Dict) -> bool:
    """ Check whether component_tensor == input_tensor + another layer's output """
    if component_tensor.shape != input_tensor.shape:
        return False
    
    residual_tensor = component_tensor - input_tensor
    slice_dim1 = min(COMPARE_LENGTH, residual_tensor.size(-2))
    slice_dim2 = min(COMPARE_LENGTH, residual_tensor.size(-1))
    for tensor in get_layer_output_tensor.values():
        if tensor.shape == residual_tensor.shape:
            if torch.allclose(tensor[..., :slice_dim1, :slice_dim2], 
                              residual_tensor[..., :slice_dim1, :slice_dim2], 
                              atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
    return False


def check_tensor_residual(input_tensor: Tensor, 
                          component_tensor: Tensor, 
                          get_layer_input_tensor: Dict) -> bool:
    """ Check whether component tensor == input_tensor + another layer's input """
    if component_tensor.shape != input_tensor.shape:
        return False
    
    residual_tensor = component_tensor - input_tensor
    slice_dim1 = min(COMPARE_LENGTH, residual_tensor.size(-2))
    slice_dim2 = min(COMPARE_LENGTH, residual_tensor.size(-1))
    for tensor in get_layer_input_tensor.values():
        if tensor.shape == residual_tensor.shape:
            if torch.allclose(tensor[..., :slice_dim1, :slice_dim2], 
                              residual_tensor[..., :slice_dim1, :slice_dim2], 
                              atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
    return False


def check_tensor_squeeze_excitation(input_tensor: Tensor,
                                    component_tensor: Tensor,
                                    get_layer_output_tensor: Dict):
    """ Check whether component tensor == input_tensor * another layer's input """
    if component_tensor.shape != input_tensor.shape:
        if (component_tensor.ndim == input_tensor.ndim 
                and component_tensor.shape[0:2] == input_tensor.shape[0:2]
                and input_tensor.shape[2:4] == torch.Size([1, 1])):
            # cause tensor multiply will automatically broadcast
            # Here we will also expand dimension for comparision
            input_tensor = input_tensor.expand_as(component_tensor)
        else:
            return False
        input_tensor = input_tensor.reshape(component_tensor.shape)
    residual_tensor = component_tensor / input_tensor
    slice_dim1 = min(COMPARE_LENGTH, residual_tensor.size(-2))
    slice_dim2 = min(COMPARE_LENGTH, residual_tensor.size(-1))
    for tensor in get_layer_output_tensor.values():
        if tensor.shape == residual_tensor.shape:
            if torch.allclose(tensor[..., :slice_dim1, :slice_dim2], 
                              residual_tensor[..., :slice_dim1, :slice_dim2], 
                              atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
        elif (tensor.ndim == residual_tensor.ndim
                and tensor.shape[0:2] == residual_tensor.shape[0:2]
                and tensor.shape[2:4] == torch.Size([1, 1])):
            # Here again, we should also expand dimension
            tensor = tensor.expand_as(residual_tensor)
            if torch.allclose(tensor[..., :slice_dim1, :slice_dim2], 
                              residual_tensor[..., :slice_dim1, :slice_dim2], 
                              atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
            
    return False


def check_tensor_in_cluster(input_tensor: Tensor, 
                            component_tensor: Tensor, 
                            cur_cluster_tensors: List):
    if component_tensor.shape != input_tensor.shape:
        if (component_tensor.ndim == input_tensor.ndim 
                and component_tensor.shape[0:2] == input_tensor.shape[0:2]
                and input_tensor.shape[2:4] == torch.Size([1, 1])):
            # again, expand input tensor if necessary
            input_tensor = input_tensor.expand_as(component_tensor)
        else:
            return False
    target_tensor1 = component_tensor - input_tensor
    target_tensor2 = component_tensor / input_tensor
    slice_dim1 = min(COMPARE_LENGTH, target_tensor1.size(-2), target_tensor2.size(-2))
    slice_dim2 = min(COMPARE_LENGTH, target_tensor1.size(-1), target_tensor2.size(-1))
    for cluster_input_tensor in cur_cluster_tensors:
        if cluster_input_tensor.shape == target_tensor1.shape:
            if torch.allclose(cluster_input_tensor[..., :slice_dim1, :slice_dim2],
                              target_tensor1[..., :slice_dim1, :slice_dim2], 
                              atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
            if torch.allclose(cluster_input_tensor[..., :slice_dim1, :slice_dim2], 
                              target_tensor2[..., :slice_dim1, :slice_dim2], 
                              atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
        elif (cluster_input_tensor.ndim == target_tensor1.ndim
                and cluster_input_tensor.shape[0:2] == target_tensor1.shape[0:2]
                and cluster_input_tensor.shape[2:4] == torch.Size([1, 1])):
            # Here again, we should also expand dimension
            cluster_input_tensor = cluster_input_tensor.expand_as(target_tensor1)
            if torch.allclose(cluster_input_tensor[..., :slice_dim1, :slice_dim2],
                              target_tensor1[..., :slice_dim1, :slice_dim2], 
                              atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
            if torch.allclose(cluster_input_tensor[..., :slice_dim1, :slice_dim2], 
                              target_tensor2[..., :slice_dim1, :slice_dim2], 
                              atol=TENSOR_DIFFERENCE_THRESHOLD):
                return True
            
    return False


def adjust_prune_distribution(prunable_layers: List,
                              prune_distribution: Tensor, 
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
    prune_distribution /= torch.sum(prune_distribution)

    """ Adjust so that layer has only 1 out dim layer cant be pruned """
    for idx, layer in enumerate(prunable_layers):
        if isinstance(layer, CONV_LAYERS) and layer.out_channels <= 1:
            prune_distribution[idx] = 0
    
    """ Adjust so that grouped convolution layers will be skiped """
    for idx, layer in enumerate(prunable_layers):
        if isinstance(layer, CONV_LAYERS) and layer.groups > 1:
            prune_distribution[idx] = 0
    return prune_distribution
