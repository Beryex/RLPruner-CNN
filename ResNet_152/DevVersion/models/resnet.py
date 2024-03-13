import torch
import torch.nn as nn

# define model parameters
prune_probability = 1
kernel_neuron_proportion = 0.33
neuron_proportion = 0.5
update_activitation_probability = 0.5
max_modification_num = 500


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
    def prune_mediate_kernel(self):
        layer_choices = torch.tensor([0, 3])
        target_layer = torch.randint(0, len(layer_choices), (1,)).item()
        target_layer = layer_choices[target_layer]
        # only need to prune self.residual_function
        new_conv1_kernel_num = self.residual_function[target_layer].out_channels - 1
        if new_conv1_kernel_num == 0:
            return 
        if target_layer == 3:
            new_conv1 = nn.Conv2d(self.residual_function[target_layer].in_channels, new_conv1_kernel_num, stride=self.residual_function[target_layer].stride, kernel_size=self.residual_function[target_layer].kernel_size, padding=1, bias=False)
        else:
            new_conv1 = nn.Conv2d(self.residual_function[target_layer].in_channels, new_conv1_kernel_num, kernel_size=self.residual_function[target_layer].kernel_size, bias=False)
        weight_variances = torch.var(self.residual_function[target_layer].weight.data, dim = [1, 2, 3])
        target_kernel = torch.argmin(weight_variances).item()
        with torch.no_grad():
            new_conv1.weight.data = torch.cat([self.residual_function[target_layer].weight.data[:target_kernel], self.residual_function[target_layer].weight.data[target_kernel+1:]], dim=0)
        self.residual_function[target_layer] = new_conv1

        new_bn1 = nn.BatchNorm2d(new_conv1_kernel_num)
        with torch.no_grad():
            kept_indices = [i for i in range(self.residual_function[target_layer + 1].num_features) if i != target_kernel]
            new_bn1.weight.data = self.residual_function[target_layer + 1].weight.data[kept_indices]
            new_bn1.bias.data = self.residual_function[target_layer + 1].bias.data[kept_indices]
            new_bn1.running_mean = self.residual_function[target_layer + 1].running_mean[kept_indices]
            new_bn1.running_var = self.residual_function[target_layer + 1].running_var[kept_indices]
        self.residual_function[target_layer + 1] = new_bn1

        # we also need to decrement next conv layer's input inside blocks
        if target_layer + 3 == 3:
            new_conv2 = nn.Conv2d(self.residual_function[target_layer + 3].in_channels - 1, self.residual_function[target_layer + 3].out_channels, stride=self.residual_function[target_layer + 3].stride, kernel_size=self.residual_function[target_layer + 3].kernel_size, padding=1, bias=False)
        else:
            new_conv2 = nn.Conv2d(self.residual_function[target_layer + 3].in_channels - 1, self.residual_function[target_layer + 3].out_channels, kernel_size=self.residual_function[target_layer + 3].kernel_size, bias=False)
        with torch.no_grad():
            kept_indices = [i for i in range(self.residual_function[target_layer + 3].in_channels) if i != target_kernel]
            new_conv2.weight.data = self.residual_function[target_layer + 3].weight.data[:, kept_indices, :, :]
        self.residual_function[target_layer + 3] = new_conv2
        
        return 
    
    def prune_output_kernel(self):
        # need to prune self.residual_function and self.shortcut at the same time
        target_layer = 6
        new_conv1_kernel_num = self.residual_function[target_layer].out_channels - 1
        if new_conv1_kernel_num == 0:
            return
        new_conv1 = nn.Conv2d(self.residual_function[target_layer].in_channels, new_conv1_kernel_num, kernel_size=self.residual_function[target_layer].kernel_size, bias=False)
        weight_variances = torch.var(self.residual_function[target_layer].weight.data, dim = [1, 2, 3])
        target_kernel = torch.argmin(weight_variances).item()
        with torch.no_grad():
            new_conv1.weight.data = torch.cat([self.residual_function[target_layer].weight.data[:target_kernel], self.residual_function[target_layer].weight.data[target_kernel+1:]], dim=0)
        self.residual_function[target_layer] = new_conv1

        new_bn1 = nn.BatchNorm2d(new_conv1_kernel_num)
        with torch.no_grad():
            kept_indices = [i for i in range(self.residual_function[target_layer + 1].num_features) if i != target_kernel]
            new_bn1.weight.data = self.residual_function[target_layer + 1].weight.data[kept_indices]
            new_bn1.bias.data = self.residual_function[target_layer + 1].bias.data[kept_indices]
            new_bn1.running_mean = self.residual_function[target_layer + 1].running_mean[kept_indices]
            new_bn1.running_var = self.residual_function[target_layer + 1].running_var[kept_indices]
        self.residual_function[target_layer + 1] = new_bn1

        if len(list(self.shortcut.children())) != 0:
            # use the same target kernel as self.residual_function
            new_conv2 = nn.Conv2d(self.shortcut[0].in_channels, new_conv1_kernel_num, stride=self.shortcut[0].stride, kernel_size=self.shortcut[0].kernel_size, bias=False)
            with torch.no_grad():
                new_conv2.weight.data = torch.cat([self.shortcut[0].weight.data[:target_kernel], self.shortcut[0].weight.data[target_kernel+1:]], dim=0)
            self.shortcut[0] = new_conv2
            new_bn2 = nn.BatchNorm2d(new_conv1_kernel_num)
            with torch.no_grad():
                kept_indices = [i for i in range(self.shortcut[1].num_features) if i != target_kernel]
                new_bn2.weight.data = self.shortcut[1].weight.data[kept_indices]
                new_bn2.bias.data = self.shortcut[1].bias.data[kept_indices]
                new_bn2.running_mean = self.shortcut[1].running_mean[kept_indices]
                new_bn2.running_var = self.shortcut[1].running_var[kept_indices]
            self.shortcut[1] = new_bn2
        # call decre_input in ResNet to prune next layer's input
        return target_kernel
    
    def decre_input(self, target_kernel):
        new_conv1 = nn.Conv2d(self.residual_function[0].in_channels - 1, self.residual_function[0].out_channels, kernel_size=self.residual_function[0].kernel_size, bias=False)
        with torch.no_grad():
            kept_indices = [i for i in range(self.residual_function[0].in_channels) if i != target_kernel]
            new_conv1.weight.data = self.residual_function[0].weight.data[:, kept_indices, :, :]
        self.residual_function[0] = new_conv1

        # if self.shortcut is empty, directly return
        if len(list(self.shortcut.children())) == 0:
            return
        new_conv2 = nn.Conv2d(self.shortcut[0].in_channels - 1, self.shortcut[0].out_channels, stride=self.shortcut[0].stride, kernel_size=self.shortcut[0].kernel_size, bias=False)
        with torch.no_grad():
            kept_indices = [i for i in range(self.shortcut[0].in_channels) if i != target_kernel]
            new_conv2.weight.data = self.shortcut[0].weight.data[:, kept_indices, :, :]
        self.shortcut[0] = new_conv2

class ResNet(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(BottleNeck, 64, 3, 1)
        self.conv3_x = self._make_layer(BottleNeck, 128, 8, 2)
        self.conv4_x = self._make_layer(BottleNeck, 256, 36, 2)
        self.conv5_x = self._make_layer(BottleNeck, 512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleNeck.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def update_architecture(self):
        update_times = torch.randint(low=int(max_modification_num / 1.5), high=max_modification_num + 1, size=(1,))
        decre_num = 0
        if torch.rand(1).item() < prune_probability:
            for update_id in range(update_times):
                if decre_num > 0:
                    decre_num -= 1
                    continue
                if torch.rand(1).item() < 0.02:
                    self.prune_kernel()
                if torch.rand(1).item() > 0.02:
                    if torch.rand(1).item() <= 5 / 6:
                        self.prune_mediate_blocks()
                    else:
                        decre_num = self.prune_output_blocks()

    def prune_kernel(self):
        target_branch = self.conv1
        target_layer = 0
        
        # prune target branch
        new_conv1_kernel_num = target_branch[target_layer].out_channels - 1
        if new_conv1_kernel_num == 0:
            return
        new_conv1 = nn.Conv2d(target_branch[target_layer].in_channels, new_conv1_kernel_num, kernel_size=target_branch[target_layer].kernel_size, padding=1, bias=False)
        weight_variances = torch.var(target_branch[target_layer].weight.data, dim = [1, 2, 3])
        target_kernel = torch.argmin(weight_variances).item()
        with torch.no_grad():
            new_conv1.weight.data = torch.cat([target_branch[target_layer].weight.data[:target_kernel], target_branch[target_layer].weight.data[target_kernel+1:]], dim=0)
        target_branch[target_layer] = new_conv1

        new_bn1 = nn.BatchNorm2d(new_conv1_kernel_num)
        with torch.no_grad():
            kept_indices = [i for i in range(target_branch[target_layer + 1].num_features) if i != target_kernel]
            new_bn1.weight.data = target_branch[target_layer + 1].weight.data[kept_indices]
            new_bn1.bias.data = target_branch[target_layer + 1].bias.data[kept_indices]
            new_bn1.running_mean = target_branch[target_layer + 1].running_mean[kept_indices]
            new_bn1.running_var = target_branch[target_layer + 1].running_var[kept_indices]
        target_branch[target_layer + 1] = new_bn1

        # we also need to modify first layer's input in conv2_x
        self.conv2_x[0].decre_input(target_kernel)

    def prune_mediate_blocks(self):
        target_block = torch.randint(1, 51, (1,)).item()
        if target_block <= 3:
            target_block = self.conv2_x[target_block - 1]
        elif target_block <= 3 + 8:
            target_block = self.conv3_x[target_block - 3 - 1]
        elif target_block <= 3 + 8 + 36:
            target_block = self.conv4_x[target_block - 8 - 3 - 1]
        else:
            target_block = self.conv5_x[target_block - 36 - 8 - 3 - 1]
            
        target_block.prune_mediate_kernel()
    
    def prune_output_blocks(self):
        # we must prune all output conv2d layer at the same time
        target_branch = torch.randint(1, 51, (1,)).item()
        if target_branch <= 3:
            target_branch = self.conv2_x
            next_branch = self.conv3_x
        elif target_branch <= 3 + 8:
            target_branch = self.conv3_x
            next_branch = self.conv4_x
        elif target_branch <= 3 + 8 + 36:
            target_branch = self.conv4_x
            next_branch = self.conv5_x
        else:
            target_branch = self.conv5_x
            next_branch = None
        
        for block_index in range(len(list(target_branch.children()))):
            target_block = target_branch[block_index]
            target_kernel = target_block.prune_output_kernel()
            next_block_idx = block_index + 1
            if next_block_idx == len(list(target_branch.children())):
                break
            next_block = target_branch[next_block_idx]
            next_block.decre_input(target_kernel)
        
        if next_branch != None:
            next_block = next_branch[0]
            next_block.decre_input(target_kernel)
        else:
            # decre FC input
            new_fc1_intput_features = self.fc.in_features - 1
            # update fc1
            new_fc1 = nn.Linear(new_fc1_intput_features, self.fc.out_features)
            output_length = 1 # gained by printing "output"
            # prune the neuron with least variance weights
            with torch.no_grad():
                start_index = target_kernel * output_length ** 2
                end_index = start_index + output_length ** 2
                new_fc1.weight.data = torch.cat([self.fc.weight.data[:, :start_index], self.fc.weight.data[:, end_index:]], dim=1)
                new_fc1.bias.data = self.fc.bias.data
            self.fc = new_fc1
        print(len(list(target_branch.children())))
        return len(list(target_branch.children()))
    
