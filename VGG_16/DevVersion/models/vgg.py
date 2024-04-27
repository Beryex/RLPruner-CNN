import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.cluster import KMeans
import faiss


class Custom_Conv2d(nn.Module):
    def __init__(self, in_channels = None, out_channels = None, kernel_size = None, stride = 1, padding = 0, bias = True, base_conv_layer = None):
        super(Custom_Conv2d, self).__init__()
        if base_conv_layer == None:
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
            if base_conv_layer.bias != None:
                self.bias = nn.Parameter(base_conv_layer.bias.data.detach())
            else:
                self.bias = None
            self.stride = base_conv_layer.stride
            self.padding = base_conv_layer.padding
        
        self.weight_indices = None
    
    def __repr__(self):
        return f'Custom_Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, weight_shape={self.weight_shape})'
    
    def forward(self, x):
        if self.weight_indices == None:
            actual_weight = self.weight
        else:
            actual_weight = self.weight[self.weight_indices].view(self.weight_shape)
        actual_weight = actual_weight.to(x.device)
        return F.conv2d(x, actual_weight, self.bias, self.stride, self.padding)


    def prune_kernel(self):
        if self.weight_indices == None:
            # weight is still normal, not cluster
            if self.out_channels - 1 == 0:
                return None
            weight_variances = torch.var(self.weight.data, dim = [1, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            with torch.no_grad():
                self.weight.data = torch.cat([self.weight.data[:target_kernel], self.weight.data[target_kernel+1:]], dim=0)
                if self.bias != None:
                    self.bias.data = torch.cat([self.bias.data[:target_kernel], self.bias.data[target_kernel+1:]])
            self.out_channels -= 1
            self.weight_shape = self.weight.shape
            if self.out_channels != self.weight.shape[0] or self.weight_shape[0] != self.weight.shape[0]:
                raise ValueError("Conv2d layer out_channels and weight dimension mismath")
            return target_kernel
    
    def decre_input(self, target_kernel):
        with torch.no_grad():
            kept_indices = [i for i in range(self.in_channels) if i != target_kernel]
            self.weight.data = self.weight.data[:, kept_indices, :, :]
        self.in_channels -= 1
        self.weight_shape = self.weight.shape
        if self.in_channels != self.weight.shape[1] or self.weight_shape[1] != self.weight.shape[1]:
            raise ValueError("Conv2d layer in_channels and weight dimension mismath")
    
    def quantization_hash_weights(self):
        original_weights = self.weight.data.clone().view(-1, 1).cuda()
        num_clusters = min(len(original_weights) // 2, 2048)

        kmeans = faiss.Kmeans(d=1, k=num_clusters, niter=50, verbose=True, gpu=True)
        kmeans.train(original_weights.cpu().numpy())

        clustered_weights = torch.from_numpy(kmeans.centroids).float().cuda().squeeze()
        labels = torch.from_numpy(kmeans.index.assign(original_weights.cpu().numpy(), k=1)).long().cuda()

        self.weight_indices = labels
        self.weight = torch.nn.Parameter(clustered_weights)

class Custom_Linear(nn.Module):
    def __init__(self, in_features = None, out_features = None, bias = True, base_linear_layer = None):
        super(Custom_Linear, self).__init__()
        if base_linear_layer == None:
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

        self.weight_indices = None
    
    def __repr__(self):
        return f'Custom_Linear(in_features={self.in_features}, out_features={self.out_features}, weight_shape={self.weight_shape})'
    
    def forward(self, x):
        if self.weight_indices == None:
            actual_weight = self.weight
        else:
            actual_weight = self.weight[self.weight_indices].view(self.weight_shape)
        actual_weight = actual_weight.to(x.device)
        return F.linear(x, actual_weight, self.bias)
    

    def prune_neuron(self):
        if self.weight_indices == None:
            # weight is still normal, not cluster
            if self.out_features - 1 == 0:
                return None
            weight_variances = torch.var(self.weight.data, dim = 1)
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                self.weight.data = torch.cat([self.weight.data[:target_neuron], self.weight.data[target_neuron+1:]], dim=0)
                if self.bias != None:
                    self.bias.data = torch.cat([self.bias.data[:target_neuron], self.bias.data[target_neuron+1:]])
            self.out_features -= 1
            self.weight_shape = self.weight.shape
            if self.out_features != self.weight.shape[0] or self.weight_shape[0] != self.weight.shape[0]:
                raise ValueError("Linear layer out_channels and weight dimension mismath")
            return target_neuron
    
    def decre_input(self, new_in_features, start_index, end_index):
        with torch.no_grad():
            self.weight.data = torch.cat([self.weight.data[:, :start_index], self.weight.data[:, end_index:]], dim=1)
            self.bias.data = self.bias.data
        self.in_features = new_in_features
        self.weight_shape = self.weight.shape
        if self.in_features != self.weight.shape[1] or self.weight_shape[1] != self.weight.shape[1]:
            raise ValueError("Linear layer in_channels and weight dimension mismath")
    
    def quantization_hash_weights(self):
        original_weights = self.weight.data.clone().view(-1, 1).cuda()
        num_clusters = min(len(original_weights) // 2, 2048)

        kmeans = faiss.Kmeans(d=1, k=num_clusters, niter=50, verbose=True, gpu=True)
        kmeans.train(original_weights.cpu().numpy())

        clustered_weights = torch.from_numpy(kmeans.centroids).float().cuda().squeeze()
        labels = torch.from_numpy(kmeans.index.assign(original_weights.cpu().numpy(), k=1)).long().cuda()

        self.weight_indices = labels
        self.weight = torch.nn.Parameter(clustered_weights)


class VGG(nn.Module):
    def __init__(self, num_class=100):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Custom_Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Custom_Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Custom_Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Custom_Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Custom_Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Custom_Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Custom_Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Custom_Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            Custom_Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Custom_Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Custom_Linear(4096, num_class)
        )

        self.prune_choices_num = 15
        self.prune_choices = torch.tensor([0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 0, 3])
        self.prune_probability = torch.tensor([0.002, 0.002, 0.002, 0.002, 0.002, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.25, 0.25])

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_layers(x)
        return x


    # define the function to resize the architecture kernel number
    def update_architecture(self, modification_num, strategy, noise_var=0.01, probability_lower_bound=0.005):
        prune_counter = torch.zeros(self.prune_choices_num)
        noise = torch.randn(self.prune_choices_num) * noise_var
        noised_distribution = self.prune_probability + noise
        noised_distribution = torch.clamp(noised_distribution, min=probability_lower_bound)
        noised_distribution = noised_distribution / torch.sum(noised_distribution)
        prune_counter = noised_distribution * modification_num
        if strategy == 'prune':
            for target_layer_idx, count in enumerate(prune_counter):
                target_layer = self.prune_choices[target_layer_idx].item()
                count = int(count.item())
                for _ in range(count):
                    if target_layer_idx < 13:
                        self.prune_conv(target_layer)
                    else:
                        self.prune_linear(target_layer)
        elif strategy == 'quantize':
            for update_id in range(modification_num):
                if torch.rand(1).item() < 0.5:
                    self.quantize_conv()
                else:
                    self.quantize_linear()
        return noised_distribution
    
    
    def update_prune_probability_distribution(self,top1_pretrain_accuracy_tensors, prune_probability_distribution_tensors, step_length, probability_lower_bound, ppo_clip):
        distribution_weight = (top1_pretrain_accuracy_tensors - torch.min(top1_pretrain_accuracy_tensors)) / torch.sum(top1_pretrain_accuracy_tensors - torch.min(top1_pretrain_accuracy_tensors))
        distribution_weight = distribution_weight.unsqueeze(1)
        weighted_distribution = torch.sum(prune_probability_distribution_tensors * distribution_weight, dim=0)
        new_prune_probability = self.prune_probability + step_length * (weighted_distribution - self.prune_probability)
        ratio = new_prune_probability / self.prune_probability
        clipped_ratio = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
        self.prune_probability *= clipped_ratio
        self.prune_probability = torch.clamp(self.prune_probability, min=probability_lower_bound)
        self.prune_probability /= torch.sum(self.prune_probability)


    def prune_conv(self, target_layer):
        # prune kernel
        target_kernel = self.conv_layers[target_layer].prune_kernel()
        if target_kernel == None:
            return

        # update bn1
        target_bn = self.conv_layers[target_layer + 1]
        with torch.no_grad():
            kept_indices = [i for i in range(target_bn.num_features) if i != target_kernel]
            target_bn.weight.data = target_bn.weight.data[kept_indices]
            target_bn.bias.data = target_bn.bias.data[kept_indices]
            target_bn.running_mean = target_bn.running_mean[kept_indices]
            target_bn.running_var = target_bn.running_var[kept_indices]
        target_bn.num_features -= 1
        if target_bn.num_features != target_bn.weight.shape[0]:
            raise ValueError("BatchNorm layer number_features and weight dimension mismath")

        if target_layer < 40:
            target_layer += 1
            while (not isinstance(self.conv_layers[target_layer], Custom_Conv2d)):
                target_layer += 1
            self.conv_layers[target_layer].decre_input(target_kernel)
        else:
            # update first FC layer
            output_length = 1 # gained by printing "output"
            new_in_features = self.conv_layers[target_layer].out_channels * output_length ** 2
            start_index = target_kernel * output_length ** 2
            end_index = start_index + output_length ** 2
            self.linear_layers[0].decre_input(new_in_features, start_index, end_index)

    def prune_linear(self, target_layer):
        target_neuron = self.linear_layers[target_layer].prune_neuron()
        if target_neuron == None:
            return
        
        new_in_features = self.linear_layers[target_layer].out_features
        start_index = target_neuron
        end_index = target_neuron + 1
        self.linear_layers[target_layer + 3].decre_input(new_in_features, start_index, end_index)
    
    def quantize_conv(self):
        if torch.rand(1).item() < 0.3:
            # low probabilit to quantize sensitive layer
            layer_choices =  torch.tensor([0, 3, 7, 10])
        else:
            layer_choices =  torch.tensor([14, 17, 20, 24, 27, 30, 34, 37, 40])
        target_layer = torch.randint(0, len(layer_choices), (1,)).item()
        target_layer = layer_choices[target_layer].item()

        # prune kernel
        self.conv_layers[target_layer].quantization_hash_weights()
    
    def quantize_linear(self):
        layer_choices =  torch.tensor([0, 3])
        target_layer = torch.randint(0, len(layer_choices), (1,)).item()
        target_layer = layer_choices[target_layer].item()

        self.linear_layers[target_layer].quantization_hash_weights()
