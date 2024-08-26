import torch
from torch import Tensor
from typing import Tuple, List
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import wandb

from conf import settings
from utils import (adjust_prune_distribution, CONV_LAYERS, NORM_LAYERS, set_inplace_false, 
                   extract_prunable_layers_info, extract_prunable_layer_dependence,
                   recover_inplace_status)


PRUNE_STRATEGY = ["variance", "l1", "l2", "activation"]


class RL_Pruner():
    """ Agent used to prune architecture and maintain prune_distribution """
    def __init__(self,
                 model: nn.Module,
                 sample_input: Tensor,
                 sample_num: int,
                 prune_filter_ratio: float,
                 noise_var: float):
        """ Extract prunable layer and layer dependence """
        logging.info(f'Start extracting layers dependency')
        print(f"Start extracting layers dependency")
        original_inplace_state = {}
        set_inplace_false(model, original_inplace_state)
        prune_distribution, filter_num, prunable_layers = extract_prunable_layers_info(model)
        next_layers, layer_cluster_mask = extract_prunable_layer_dependence(model, 
                                                                            sample_input, 
                                                                            prunable_layers)
        assert len(prunable_layers) == len(next_layers) == prune_distribution.shape[0]
        recover_inplace_status(model, original_inplace_state)
        prune_distribution = adjust_prune_distribution(prunable_layers,
                                                       prune_distribution, 
                                                       layer_cluster_mask)
        logging.info(f'Complete extracting layers dependency')
        print(f"Complete extracting layers dependency")
        for i in range(len(prune_distribution)):
            wandb.log({f"prune_distribution_item_{i}": prune_distribution[i]}, 
                       step=0)

        """ Static Data: These attributes do not change after initialization """
        self.sample_num = sample_num
        self.modification_num = int(filter_num * prune_filter_ratio)
        self.layer_cluster_mask = layer_cluster_mask
        self.noise_var = noise_var
        
        """ Dynamic Data: These attributes may change during the object's lifetime """
        self.model = model
        self.prunable_layers = prunable_layers
        self.next_layers = next_layers
        self.prune_distribution = prune_distribution
        self.all_layer_filter_importance = None
        # Replay buffer [:, 0] stores Q value Q(s, a), [:, 1:] stores action PD
        self.ReplayBuffer = torch.zeros([sample_num, 1 + len(prune_distribution)])
        self.model_info_list = [None] * sample_num


    def link_model(self, model_with_info: Tuple) -> None:
        """ Link the model in class with input model with info """
        self.model = model_with_info[0]
        self.prunable_layers = model_with_info[1]
        self.next_layers = model_with_info[2]
    

    def reinitialize_PD(self) -> None:
        """ Reinitialize prune distribution to be uniform """
        for idx, layer in enumerate(self.prunable_layers):
            if isinstance(layer, CONV_LAYERS):
                self.prune_distribution[idx] = layer.out_channels
            elif isinstance(layer, nn.Linear):
                self.prune_distribution[idx] = layer.out_features
        self.prune_distribution /= torch.sum(self.prune_distribution)
        self.prune_distribution = adjust_prune_distribution(self.prunable_layers,
                                                            self.prune_distribution, 
                                                            self.layer_cluster_mask)
    

    def resume_model(self, model: nn.Module, sample_input: Tensor) -> None:
        """ Resume the model and link it """
        logging.info(f'Start extracting layers dependency')
        print(f"Start extracting layers dependency")
        original_inplace_state = {}
        set_inplace_false(model, original_inplace_state)
        _, _, prunable_layers = extract_prunable_layers_info(model)
        next_layers, _ = extract_prunable_layer_dependence(model, 
                                                           sample_input, 
                                                           prunable_layers)
        assert len(prunable_layers) == len(next_layers) == self.prune_distribution.shape[0]
        recover_inplace_status(model, original_inplace_state)
        logging.info(f'Complete extracting layers dependency')
        print(f"Complete extracting layers dependency")

        resumed_model_with_info = (model, prunable_layers, next_layers)
        self.link_model(resumed_model_with_info)
    

    def get_linked_model(self) -> Tuple:
        """ Get the linked model associated with current prune agent """
        model_with_info = (self.model, self.prunable_layers, self.next_layers)
        return model_with_info


    def clear_cache(self) -> None:
        """ Clear the ReplayBuffer and model_info_list """
        self.ReplayBuffer.zero_()
        self.model_info_list = [None] * self.sample_num


    @torch.no_grad()
    def update_prune_distribution(self, 
                                  step_length: float,  
                                  ppo_clip: float, 
                                  ppo_enable: bool) -> Tensor:
        """ Update prune distribution and return its change """
        P_lower_bound = settings.RL_PROBABILITY_LOWER_BOUND
        original_PD = self.prune_distribution
        _, optimal_idx = torch.max(self.ReplayBuffer[:, 0], dim=0)
        optimal_PD = self.ReplayBuffer[optimal_idx, 1:]
        
        updated_PD = original_PD + step_length * (optimal_PD - original_PD) 
        updated_PD = torch.clamp(updated_PD, min=P_lower_bound)
        updated_PD = adjust_prune_distribution(self.prunable_layers,
                                               updated_PD, 
                                               self.layer_cluster_mask)
        
        if ppo_enable == True:
            # apply PPO to make PD changes stably
            original_PD[original_PD == 0] = 1e-6 # in case of devide by 0
            ratio = updated_PD / original_PD
            updated_PD = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * original_PD
            updated_PD = torch.clamp(updated_PD, min=P_lower_bound)
            updated_PD = adjust_prune_distribution(self.prunable_layers,
                                                   updated_PD, 
                                                   self.layer_cluster_mask)
        self.prune_distribution = updated_PD
        return updated_PD - original_PD


    @torch.no_grad()
    def prune_architecture(self,
                           eval_loader: DataLoader,
                           prune_strategy: str) -> Tensor:
        """ Generate new noised PD and prune architecture based on noised PD """
        P_lower_bound = settings.RL_PROBABILITY_LOWER_BOUND
        prune_counter = torch.zeros(len(self.prunable_layers))
        noise = torch.randn(len(self.prune_distribution)) * self.noise_var * torch.rand(1).item()
        noised_PD = self.prune_distribution + noise
        noised_PD = torch.clamp(noised_PD, min=P_lower_bound)
        noised_PD = noised_PD / torch.sum(noised_PD)
        noised_PD = adjust_prune_distribution(self.prunable_layers, noised_PD, self.layer_cluster_mask)
        prune_counter = torch.round(noised_PD * self.modification_num).int().tolist()

        """ Get each filter's importance """
        if "activation" == prune_strategy:
            all_layer_filter_importance = self.get_filter_importance_activation(eval_loader)
        else:
            all_layer_filter_importance = self.get_filter_importance_weight(prune_strategy)
        self.all_layer_filter_importance = all_layer_filter_importance
            
        """ Average the filter importance inside a cluster to represent overall importance """
        cluster_filter_importance = {}  # index 0 stores filter importance tensor, index 1 stores layer number
        for target_layer_idx, count in enumerate(prune_counter):
            cluster_mask = self.layer_cluster_mask[target_layer_idx]
            if cluster_mask > 0:
                filter_importance = all_layer_filter_importance[target_layer_idx]
                if cluster_mask not in cluster_filter_importance:
                    cluster_filter_importance[cluster_mask] = [filter_importance, 1]
                else:
                    if len(filter_importance) != len(cluster_filter_importance[cluster_mask][0]):
                        raise ValueError(f"The filter numbers inside a cluster are different")
                    cluster_filter_importance[cluster_mask] += [filter_importance, 1]
        
        for target_layer_idx, count in enumerate(prune_counter):
            cluster_mask = self.layer_cluster_mask[target_layer_idx]
            if cluster_mask > 0:
                all_layer_filter_importance[target_layer_idx] = (cluster_filter_importance[cluster_mask][0] /
                                                                 cluster_filter_importance[cluster_mask][1])
            
        """ Prune each layer's filter based on importance """
        # decred_layer: used to track which layer's next PRUNABLE_LAYER has been decred input (marked as 1)
        # in case in a cluster layer1 and layer2 combined as input to layer3, then 
        # we could decre layer3 input twice
        # we should not prevent mutiple decre input outside layer cluster
        decred_layer = torch.zeros(len(self.prunable_layers))   

        for target_layer_idx, count in enumerate(prune_counter):
            target_layer = self.prunable_layers[target_layer_idx]
            filter_importance = all_layer_filter_importance[target_layer_idx]
            
            if isinstance(target_layer, CONV_LAYERS):
                prune_filter =  self.prune_conv_filter
            elif isinstance(target_layer, (nn.Linear)):
                prune_filter = self.prune_linear_filter
            else:
                raise ValueError(f"Unsupported layer type: {target_layer}")
            
            if self.layer_cluster_mask[target_layer_idx] > 0:
                if decred_layer[target_layer_idx] == 1:
                    decre_input = False
                else:
                    decre_input = True
                    # mask all layer inside the same cluster as decred
                    target_mask = self.layer_cluster_mask[target_layer_idx]
                    for idx, mask in enumerate(self.layer_cluster_mask):
                        if mask == target_mask:
                            decred_layer[idx] = 1
            else:
                decre_input = True
            for _ in range(count):
                if len(filter_importance) > 1:
                    target_filter_idx = torch.argmin(filter_importance).item()
                    filter_importance = torch.cat((filter_importance[:target_filter_idx], 
                                                    filter_importance[target_filter_idx + 1:]))
                    prune_filter(target_layer_idx, 
                                 target_layer, 
                                 target_filter_idx,
                                 prune=True,
                                 decre_input=decre_input)
                else:
                    break

        return noised_PD


    def get_filter_importance_activation(self,
                                         eval_loader: DataLoader) -> List:
        """ Compute the importance score of all filters based on its activation """
        all_layer_filter_importance = []
        handles = []
        get_layer_outputs = {}

        def forward_hook(layer: nn.Module, input: Tuple, output: Tensor) -> None:
            """ Track each layer's output """
            if layer in self.prunable_layers:
                if layer in get_layer_outputs:
                    get_layer_outputs[layer] += output.sum(dim=0)
                else:
                    get_layer_outputs[layer] = output.sum(dim=0)

        for layer in self.prunable_layers:
            handle = layer.register_forward_hook(forward_hook)
            handles.append(handle)
        
        device = next(self.model.parameters()).device
        self.model.eval()
        for images, _ in eval_loader:
            images = images.to(device)
            self.model(images)

        for handle in handles:
            handle.remove()
        
        for target_layer in self.prunable_layers:
            if isinstance(target_layer, CONV_LAYERS):
                all_layer_filter_importance.append(torch.sum(get_layer_outputs[target_layer] ** 2, dim = [1, 2]))
            elif isinstance(target_layer, (nn.Linear)):
                all_layer_filter_importance.append(get_layer_outputs[target_layer])
            
        return all_layer_filter_importance


    def get_filter_importance_weight(self, prune_strategy: str) -> List:
        """ Compute the importance score of all filters based on weights """
        all_layer_filter_importance = []
        for target_layer in self.prunable_layers:
            if prune_strategy == "variance":
                if isinstance(target_layer, CONV_LAYERS):
                    all_layer_filter_importance.append(torch.var(target_layer.weight.data, dim = [1, 2, 3]))
                elif isinstance(target_layer, (nn.Linear)):
                    all_layer_filter_importance.append(torch.var(target_layer.weight.data, dim = 1))
            elif prune_strategy == "l1":
                if isinstance(target_layer, CONV_LAYERS):
                    all_layer_filter_importance.append(torch.sum(torch.abs(target_layer.weight.data), dim = [1, 2, 3]))
                elif isinstance(target_layer, (nn.Linear)):
                    all_layer_filter_importance.append(torch.sum(torch.abs(target_layer.weight.data), dim = 1))
            elif prune_strategy == "l2":
                if isinstance(target_layer, CONV_LAYERS):
                    all_layer_filter_importance.append(torch.sum(target_layer.weight.data ** 2, dim = [1, 2, 3]))
                elif isinstance(target_layer, (nn.Linear)):
                    all_layer_filter_importance.append(torch.sum(target_layer.weight.data ** 2, dim = 1))
            else:
                raise ValueError(f'Unsupported prune strategy: {prune_strategy}')
        return all_layer_filter_importance


    def prune_conv_filter(self,
                          target_layer_idx: int,
                          target_layer: nn.Module, 
                          target_filter_idx: int,
                          prune: bool,
                          decre_input: bool) -> None:
        """ Prune one conv filter and decrease next layers' input dim """
        if prune == True:
            with torch.no_grad():
                target_layer.weight.data = torch.cat([target_layer.weight.data[:target_filter_idx], 
                                                    target_layer.weight.data[target_filter_idx+1:]], 
                                                    dim=0)
                if target_layer.bias is not None:
                    target_layer.bias.data = torch.cat([target_layer.bias.data[:target_filter_idx], 
                                                        target_layer.bias.data[target_filter_idx+1:]])
            target_layer.out_channels -= 1
            if target_layer.out_channels != target_layer.weight.shape[0]:
                raise ValueError(f'Conv2d layer out_channels {target_layer.out_channels} and '
                                f'weight dimension {target_layer.weight.shape[0]} mismatch')

        for next_layer_info in self.next_layers[target_layer_idx]:
            next_layer = next_layer_info[0]
            offset = max(next_layer_info[1], 0) # avoid offset to be -1 for residual connection case
            
            if isinstance(next_layer, NORM_LAYERS):
                # case 1: BatchNorm
                target_bn = next_layer
                with torch.no_grad():
                    kept_indices = [i for i in range(target_bn.num_features) 
                                    if i != target_filter_idx + offset]
                    target_bn.weight.data = target_bn.weight.data[kept_indices]
                    target_bn.bias.data = target_bn.bias.data[kept_indices]
                    target_bn.running_mean = target_bn.running_mean[kept_indices]
                    target_bn.running_var = target_bn.running_var[kept_indices]
                target_bn.num_features -= 1
                self.decrease_offset(next_layer, offset, 1)
                if target_bn.num_features != target_bn.weight.shape[0]:
                    raise ValueError(f'BatchNorm layer number_features {target_bn.num_features} and '
                                    f'weight dimension {target_bn.weight.shape[0]} mismatch')
            
            elif isinstance(next_layer, CONV_LAYERS) and next_layer.groups == 1 and decre_input:
                # case 2: Standard Conv
                with torch.no_grad():
                    kept_indices = [i for i in range(next_layer.in_channels) 
                                    if i != target_filter_idx + offset]
                    next_layer.weight.data = next_layer.weight.data[:, kept_indices, :, :]
                next_layer.in_channels -= 1
                self.decrease_offset(next_layer, offset, 1)
                if next_layer.in_channels != next_layer.weight.shape[1]:
                    raise ValueError(f'Conv2d layer in_channels {next_layer.in_channels} and '
                                    f'weight dimension {next_layer.weight.shape[1]} mismatch')
            
            elif isinstance(next_layer, CONV_LAYERS) and next_layer.groups == next_layer.in_channels and decre_input:
                # case 3: Depthwise Conv, where we only need to decre out dim, as in dim is always 1
                with torch.no_grad():
                    kept_indices = [i for i in range(next_layer.in_channels) 
                                    if i != target_filter_idx + offset]
                    next_layer.weight.data = next_layer.weight.data[kept_indices, :, :, :]
                next_layer.in_channels -= 1
                next_layer.out_channels -= 1
                next_layer.groups -= 1
                self.decrease_offset(next_layer, offset, 1)
                if next_layer.out_channels != next_layer.weight.shape[0]:
                    raise ValueError(f'Conv2d layer in_channels {next_layer.out_channels} and '
                                    f'weight dimension {next_layer.weight.shape[0]} mismatch')
                if next_layer.out_channels != next_layer.in_channels:
                    raise ValueError(f'Conv2d layer in_channels {next_layer.out_channels} and '
                                    f'weight dimension {next_layer.in_channels} mismatch')
                # we need decre next layers after this depthwise conv layer manually
                for target_layer_idx, layer in enumerate(self.prunable_layers):
                    if id(layer) == id(next_layer):
                        self.prune_conv_filter(target_layer_idx,
                                               next_layer,
                                               target_filter_idx + offset,
                                               prune=False,
                                               decre_input=True)
                        break
            
            elif isinstance(next_layer, (nn.Linear)) and decre_input:
                # case 4: Linear
                output_area = 1 # default for most CNNs
                start_index = (target_filter_idx + offset) * output_area ** 2
                end_index = start_index + output_area ** 2
                with torch.no_grad():
                    next_layer.weight.data = torch.cat([next_layer.weight.data[:, :start_index], 
                                                        next_layer.weight.data[:, end_index:]], 
                                                        dim=1)
                    if next_layer.bias is not None:
                        next_layer.bias.data = next_layer.bias.data
                next_layer.in_features -= output_area ** 2
                self.decrease_offset(next_layer, offset, output_area ** 2)
                if next_layer.in_features != next_layer.weight.shape[1]:
                    raise ValueError(f'Linear layer in_channels {next_layer.in_features} and '
                                    f'weight dimension {next_layer.weight.shape[1]} mismatch')


    def prune_linear_filter(self,
                            target_layer_idx: int,
                            target_layer: nn.Linear,
                            target_filter_idx: int,
                            prune: bool,
                            decre_input: bool) -> None:
        """ Prune one linear filter and decrease next layers' input dim """
        if prune == True:
            with torch.no_grad():
                target_layer.weight.data = torch.cat([target_layer.weight.data[:target_filter_idx], 
                                                    target_layer.weight.data[target_filter_idx+1:]], 
                                                    dim=0)
                if target_layer.bias is not None:
                    target_layer.bias.data = torch.cat([target_layer.bias.data[:target_filter_idx], 
                                                        target_layer.bias.data[target_filter_idx+1:]])
            target_layer.out_features -= 1
            if target_layer.out_features != target_layer.weight.shape[0]:
                raise ValueError(f'Linear layer out_channels {target_layer.out_features} and '
                                f'weight dimension {target_layer.weight.shape[0]} mismatch')
        
        # update following layers
        for next_layer_info in self.next_layers[target_layer_idx]:
            next_layer = next_layer_info[0]
            offset = max(next_layer_info[1], 0)
            if isinstance(next_layer, NORM_LAYERS):
                # case 1: BatchNorm
                target_bn = next_layer
                with torch.no_grad():
                    kept_indices = [i for i in range(target_bn.num_features) 
                                    if i != target_filter_idx + offset]
                    target_bn.weight.data = target_bn.weight.data[kept_indices]
                    target_bn.bias.data = target_bn.bias.data[kept_indices]
                    target_bn.running_mean = target_bn.running_mean[kept_indices]
                    target_bn.running_var = target_bn.running_var[kept_indices]
                target_bn.num_features -= 1
                self.decrease_offset(next_layer, offset, 1)
                if target_bn.num_features != target_bn.weight.shape[0]:
                    raise ValueError(f'BatchNorm layer number_features {target_bn.num_features} and '
                                    f'weight dimension {target_bn.weight.shape[0]} mismatch')
                
            elif isinstance(next_layer, (nn.Linear)) and decre_input:
                # case 2: Linear
                start_index = target_filter_idx + offset
                end_index = start_index + 1
                with torch.no_grad():
                    next_layer.weight.data = torch.cat([next_layer.weight.data[:, :start_index], 
                                                        next_layer.weight.data[:, end_index:]], 
                                                        dim=1)
                    if next_layer.bias is not None:
                        next_layer.bias.data = next_layer.bias.data
                next_layer.in_features -= 1
                self.decrease_offset(next_layer, offset, 1)
                if next_layer.in_features != next_layer.weight.shape[1]:
                    raise ValueError(f'Linear layer in_channels {next_layer.in_features} and '
                                    f'weight dimension {next_layer.weight.shape[1]} mismatch')


    def decrease_offset(self,
                        target_layer: nn.Module, 
                        target_offset: int, 
                        decrement: int) -> None:
        """ decrease offset for each next layer after we decre next layer input dim """
        # loop through all next layers is necessary because target layer could be included in 
        # multiple next layers
        for ith_next_layers in self.next_layers:
            for next_layer_info in ith_next_layers:
                next_layer = next_layer_info[0]
                offset = next_layer_info[1]
                if id(target_layer) == id(next_layer) and offset > target_offset:
                    next_layer_info[1] -= decrement
