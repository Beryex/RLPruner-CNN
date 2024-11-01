import torch
from torch import Tensor
from typing import Tuple, List, Dict
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import wandb
import math

from conf import settings
from utils import (adjust_prune_distribution, CONV_LAYERS, NORM_LAYERS, set_inplace_false, 
                   extract_prunable_layers_info, extract_prunable_layer_dependence,
                   recover_inplace_status)


PRUNE_STRATEGY = ["random", "variance", "l1", "l2", "taylor"]
EXPLORE_STRATEGY = ["constant", "linear", "cosine"]


class RL_Pruner():
    """ Agent used to prune architecture and maintain prune_distribution """
    def __init__(self,
                 model: nn.Module,
                 skip_layer_index: str,
                 sample_input: Tensor,
                 prune_steps: int,
                 explore_strategy: str,
                 greedy_epsilon: float,
                 prune_strategy: str,
                 sample_num: int,
                 prune_filter_ratio: float,
                 noise_var: float,
                 Q_FLOP_coef: float,
                 Q_Para_coef: float):
        """ Extract prunable layer and layer dependence """
        logging.info(f'Start extracting layers dependency')
        print(f"Start extracting layers dependency")
        original_inplace_state = {}
        set_inplace_false(model, original_inplace_state)
        if skip_layer_index == '':
            skip_layer_index = []
        else:
            skip_layer_index = [int(item) for item in skip_layer_index.split(',')]
        prune_distribution, filter_num, prunable_layers = extract_prunable_layers_info(model, 
                                                                                       skip_layer_index)
        logging.info(f"Skip layer at index: {skip_layer_index}")
        print(f"Skip layer at index: {skip_layer_index}")
        
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
        self.skip_layer_index = skip_layer_index
        self.prune_steps = prune_steps
        self.explore_strategy = explore_strategy
        self.initial_greedy_epsilon = greedy_epsilon
        self.prune_strategy = prune_strategy
        self.sample_num = sample_num
        self.modification_num = int(filter_num * prune_filter_ratio)
        self.layer_cluster_mask = layer_cluster_mask
        self.noise_var = noise_var
        self.Q_FLOP_coef = Q_FLOP_coef
        self.Q_Para_coef = Q_Para_coef
        
        """ Dynamic Data: These attributes may change during the object's lifetime """
        self.cur_step = 0
        self.model = model
        self.prunable_layers = prunable_layers
        self.next_layers = next_layers
        self.prune_distribution = prune_distribution
        self.all_layer_filter_importance = None
        # Replay buffer [:, 0] stores Q value Q(s, a), [:, 1:] stores action PD
        self.ReplayBuffer = torch.zeros([sample_num, 1 + len(prune_distribution)])
        self.model_info_list = [None] * sample_num
        self.greedy_epsilon = self.initial_greedy_epsilon


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
    

    def step(self) -> None:
        """ Increment prune step and update relevant parameter"""
        self.cur_step += 1
        if self.cur_step <= self.prune_steps * 0.1:
            if self.explore_strategy == "linear":
                self.greedy_epsilon = max((1 - self.cur_step / (self.prune_steps * 0.1)) * self.initial_greedy_epsilon, 0)
            elif self.explore_strategy == "cosine":
                self.greedy_epsilon = max(0.5 * (1 + math.cos(math.pi * self.cur_step / (self.prune_steps * 0.1))) * self.initial_greedy_epsilon, 0)
        else:
            self.greedy_epsilon = 0

    def resume_model(self, model: nn.Module, sample_input: Tensor) -> None:
        """ Resume the model and link it """
        logging.info(f'Start extracting layers dependency')
        print(f"Start extracting layers dependency")
        original_inplace_state = {}
        set_inplace_false(model, original_inplace_state)
        _, _, prunable_layers = extract_prunable_layers_info(model,
                                                             self.skip_layer_index)
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


    def prune_architecture(self,
                           model: nn.Module,
                           calibration_loader: DataLoader) -> Tensor:
        """ Generate new noised PD and prune architecture based on noised PD """
        P_lower_bound = settings.RL_PROBABILITY_LOWER_BOUND
        noise = torch.randn(len(self.prune_distribution)) * self.noise_var * torch.rand(1).item()
        noised_PD = self.prune_distribution + noise
        noised_PD = torch.clamp(noised_PD, min=P_lower_bound)
        noised_PD = noised_PD / torch.sum(noised_PD)
        noised_PD = adjust_prune_distribution(self.prunable_layers, noised_PD, self.layer_cluster_mask)
        prune_counter = self._get_prune_counter(noised_PD)

        """ Get each filter's importance """
        all_layer_filter_importance = self._get_filter_importance(self.prune_strategy, model, calibration_loader)
        self.all_layer_filter_importance = all_layer_filter_importance
            
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
                prune_filter =  self._prune_conv_filter
            elif isinstance(target_layer, (nn.Linear)):
                prune_filter = self._prune_linear_filter
            
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
                if len(filter_importance) > 1:  # prevent deleting the whole layer
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

    
    def Q_value_function(self, top1_acc: float, FLOPs_ratio: float, Para_ratio: float) -> float:
        """ Compute the Q value """
        return top1_acc + self.Q_FLOP_coef * FLOPs_ratio + self.Q_Para_coef * Para_ratio
    

    def update_ReplayBuffer(self, 
                            Q_value: float, 
                            prune_distribution_action: Tensor, 
                            generated_model_with_info: List) -> None:
        min_top1_acc, min_idx = torch.min(self.ReplayBuffer[:, 0], dim=0)
        if Q_value >= min_top1_acc:
            self.ReplayBuffer[min_idx, 0] = Q_value
            self.ReplayBuffer[min_idx, 1:] = prune_distribution_action
            self.model_info_list[min_idx] = generated_model_with_info
            
    
    def get_optimal_compressed_model(self) -> List:
        """ Use epsilon-greedy exploration strategy to get optimal compressed model """
        if torch.rand(1).item() < self.greedy_epsilon:
            best_model_index = torch.randint(0, self.ReplayBuffer.shape[0], (1,)).item()
            logging.info(f'Exploration: model {best_model_index} is the best new model')
        else:
            best_model_index = torch.argmax(self.ReplayBuffer[:, 0])
            logging.info(f'Exploitation: model {best_model_index} is the best new model')
        best_generated_model_with_info = self.model_info_list[best_model_index]
        return best_generated_model_with_info


    def _get_prune_counter(self, PD: Tensor) -> List:
        """ Get the number of filter to be pruned for each layer """
        prune_counter = torch.round(PD * self.modification_num).int()
        
        # Adjust if the sum doesn't match self.modification_num
        initial_sum = prune_counter.sum().item()
        difference = self.modification_num - initial_sum
        incre = True if difference > 0 else False
        dif_list = [abs(difference)]
        mod_list = [self.modification_num]
        cur_mod = self.modification_num
        while True:
            cur_mod += 1 if incre else -1
            cur_prune_counter = torch.round(PD * cur_mod).int()
            cur_sum = cur_prune_counter.sum().item()
            cur_dif = self.modification_num - cur_sum
            dif_list.append(abs(cur_dif))
            mod_list.append(cur_mod)
            if (cur_dif < 0 and incre == True) or (cur_dif > 0 and incre == False):
                break
        target_idx = dif_list.index(min(dif_list))
        target_modification_num = mod_list[target_idx]
        target_prune_counter = torch.round(PD * target_modification_num).int()
        
        target_prune_counter = target_prune_counter.tolist()
        return target_prune_counter


    def _get_filter_importance(self, 
                               prune_strategy: str,
                               model: nn.Module, 
                               calibration_loader: DataLoader) -> List:
        """ Compute the importance score of all filters based on weights """
        all_layer_filter_importance = []

        if prune_strategy == "taylor":
            model.train()
            model.zero_grad()
            device = next(model.parameters()).device
            loss_function = nn.CrossEntropyLoss()
            total_loss = 0
            
            for images, labels in calibration_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                total_loss += loss_function(outputs, labels)
            
            total_loss.backward()
            
            for target_layer in self.prunable_layers:
                if prune_strategy == "taylor":
                    if isinstance(target_layer, CONV_LAYERS):
                        salience = target_layer.weight.data * target_layer.weight.grad
                        taylor_importance = torch.sum(torch.abs(salience), dim=[1, 2, 3])
                        all_layer_filter_importance.append(taylor_importance)
                    elif isinstance(target_layer, nn.Linear):
                        salience = target_layer.weight.data * target_layer.weight.grad
                        taylor_importance = torch.sum(torch.abs(salience), dim=1)
                        all_layer_filter_importance.append(taylor_importance)
                else:
                    raise ValueError(f'Unsupported prune strategy: {prune_strategy}')
            model.zero_grad()
        
        else:
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
                
                elif prune_strategy == "random":
                    all_layer_filter_importance.append(torch.rand(target_layer.weight.size(0)))
                
                else:
                    raise ValueError(f'Unsupported prune strategy: {prune_strategy}')
        
        """ Average the filter importance inside a cluster to represent overall importance """
        cluster_filter_importance = {}  # index 0 stores filter importance tensor, index 1 stores number of layers in that cluster
        for target_layer_idx, layer in enumerate(self.prunable_layers):
            cluster_mask = self.layer_cluster_mask[target_layer_idx]
            if cluster_mask > 0:
                filter_importance = all_layer_filter_importance[target_layer_idx]
                if cluster_mask not in cluster_filter_importance:
                    cluster_filter_importance[cluster_mask] = [filter_importance, 1]
                else:
                    if len(filter_importance) != len(cluster_filter_importance[cluster_mask][0]):
                        raise ValueError(f"The filter numbers inside a cluster are different")
                    cluster_filter_importance[cluster_mask][0] += filter_importance
                    cluster_filter_importance[cluster_mask][1] += 1
        
        for target_layer_idx, layer in enumerate(self.prunable_layers):
            cluster_mask = self.layer_cluster_mask[target_layer_idx]
            if cluster_mask > 0:
                all_layer_filter_importance[target_layer_idx] = (cluster_filter_importance[cluster_mask][0] /
                                                                 cluster_filter_importance[cluster_mask][1])
                
        return all_layer_filter_importance


    @torch.no_grad()
    def _prune_conv_filter(self,
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

        for idx, next_layer_info in enumerate(self.next_layers[target_layer_idx]):
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
                self._decrease_offset(next_layer, offset, 1)
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
                self._decrease_offset(next_layer, offset, 1)
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
                self._decrease_offset(next_layer, offset, 1)
                if next_layer.out_channels != next_layer.weight.shape[0]:
                    raise ValueError(f'Conv2d layer in_channels {next_layer.out_channels} and '
                                     f'weight dimension {next_layer.weight.shape[0]} mismatch')
                if next_layer.out_channels != next_layer.in_channels:
                    raise ValueError(f'Conv2d layer in_channels {next_layer.out_channels} and '
                                     f'weight dimension {next_layer.in_channels} mismatch')
                if next_layer.out_channels != next_layer.groups:
                    raise ValueError(f'Conv2d layer in_channels {next_layer.out_channels} and '
                                     f'weight dimension {next_layer.groups} mismatch')
                # we need decre next layers after this depthwise conv layer manually
                for target_layer_idx, layer in enumerate(self.prunable_layers):
                    if id(layer) == id(next_layer):
                        self._prune_conv_filter(target_layer_idx,
                                                next_layer,
                                                target_filter_idx + offset,
                                                prune=False,
                                                decre_input=True)
                        break
            
            elif isinstance(next_layer, (nn.Linear)) and decre_input:
                # case 4: Linear
                # we need to check whether previous layer is pooling
                pre_layer = self.next_layers[target_layer_idx][idx - 1][0]
                if isinstance(pre_layer, nn.AdaptiveAvgPool2d):
                    if isinstance(pre_layer.output_size, int):
                        output_area = pre_layer.output_size
                    else:
                        output_area = pre_layer.output_size[0] * pre_layer.output_size[1]
                else:
                    output_area = 1
                start_index = (target_filter_idx + offset) * output_area
                end_index = start_index + output_area
                with torch.no_grad():
                    next_layer.weight.data = torch.cat([next_layer.weight.data[:, :start_index], 
                                                        next_layer.weight.data[:, end_index:]], 
                                                        dim=1)
                    if next_layer.bias is not None:
                        next_layer.bias.data = next_layer.bias.data
                next_layer.in_features -= output_area
                self._decrease_offset(next_layer, offset, output_area)
                if next_layer.in_features != next_layer.weight.shape[1]:
                    raise ValueError(f'Linear layer in_channels {next_layer.in_features} and '
                                    f'weight dimension {next_layer.weight.shape[1]} mismatch')


    @torch.no_grad()
    def _prune_linear_filter(self,
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
                self._decrease_offset(next_layer, offset, 1)
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
                self._decrease_offset(next_layer, offset, 1)
                if next_layer.in_features != next_layer.weight.shape[1]:
                    raise ValueError(f'Linear layer in_channels {next_layer.in_features} and '
                                    f'weight dimension {next_layer.weight.shape[1]} mismatch')


    def _decrease_offset(self,
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
