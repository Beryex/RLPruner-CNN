import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        self.branch1_out = n1x1
        self.branch2_out = n3x3
        self.branch3_out = n5x5
        self.branch4_out = pool_proj

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
    
    def prune_kernel(self):
        branch_choices = torch.tensor([1, 2, 3, 4])
        target_branch = torch.randint(0, len(branch_choices), (1,)).item()
        target_branch = branch_choices[target_branch]
        if target_branch == 1:
            target_branch = self.b1
            target_layer = 0
        elif target_branch == 2:
            target_branch = self.b2
            layer_choices = torch.tensor([0, 3])
            target_layer = torch.randint(0, len(layer_choices), (1,)).item()
            target_layer = layer_choices[target_layer]
        elif target_branch == 3:
            target_branch = self.b3
            layer_choices = torch.tensor([0, 3, 6])
            target_layer = torch.randint(0, len(layer_choices), (1,)).item()
            target_layer = layer_choices[target_layer]
        elif target_branch == 4:
            target_branch = self.b4
            target_layer = 1
        
        # prune target branch
        new_conv1_kernel_num = target_branch[target_layer].out_channels - 1
        if new_conv1_kernel_num == 0:
            return None, None
        if target_layer >= 3:
            new_conv1 = nn.Conv2d(target_branch[target_layer].in_channels, new_conv1_kernel_num, kernel_size=target_branch[target_layer].kernel_size, padding=1, bias=False)
        else:
            new_conv1 = nn.Conv2d(target_branch[target_layer].in_channels, new_conv1_kernel_num, kernel_size=target_branch[target_layer].kernel_size, bias=False)
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
        if target_branch == self.b1:
            self.branch1_out -= 1
            return 1, target_kernel
        elif target_branch == self.b2 and target_layer == 3:
            self.branch2_out -= 1
            return 2, target_kernel
        elif target_branch == self.b3 and target_layer == 6:
            self.branch3_out -= 1
            return 3, target_kernel
        elif target_branch == self.b4:
            self.branch4_out -= 1
            return 4, target_kernel
        else:
            # else, we need to decrement next conv layer's input inside inception
            if target_layer + 3 >= 3:
                new_conv2 = nn.Conv2d(target_branch[target_layer + 3].in_channels - 1, target_branch[target_layer + 3].out_channels, kernel_size=target_branch[target_layer + 3].kernel_size, padding=1, bias=False)
            else:
                new_conv2 = nn.Conv2d(target_branch[target_layer + 3].in_channels - 1, target_branch[target_layer + 3].out_channels, kernel_size=target_branch[target_layer + 3].kernel_size, bias=False)
            with torch.no_grad():
                kept_indices = [i for i in range(target_branch[target_layer + 3].in_channels) if i != target_kernel]
                new_conv2.weight.data = target_branch[target_layer + 3].weight.data[:, kept_indices, :, :]
            target_branch[target_layer + 3] = new_conv2
            return None, target_kernel
    
    def change_activation_function(self):
        branch_choices = torch.tensor([1, 2, 3, 4])
        target_branch = torch.randint(0, len(branch_choices), (1,)).item()
        target_branch = branch_choices[target_branch]
        if target_branch == 1:
            target_branch = self.b1
            target_layer = 2
        elif target_branch == 2:
            target_branch = self.b2
            layer_choices = torch.tensor([2, 5])
            target_layer = torch.randint(0, len(layer_choices), (1,)).item()
            target_layer = layer_choices[target_layer]
        elif target_branch == 3:
            target_branch = self.b3
            layer_choices = torch.tensor([2, 5, 8])
            target_layer = torch.randint(0, len(layer_choices), (1,)).item()
            target_layer = layer_choices[target_layer]
        elif target_branch == 4:
            target_branch = self.b4
            target_layer = 3
        p1 = torch.rand(1).item()
        if p1 < 0.2:
            target_branch[target_layer] = torch.nn.ReLU(inplace=True)
        elif p1 < 0.4:
            target_branch[target_layer] = torch.nn.Tanh()
        elif p1 < 0.6:
            target_branch[target_layer] = torch.nn.LeakyReLU(inplace=True)
        elif p1 < 0.8:
            target_branch[target_layer] = torch.nn.Sigmoid()
        else:
            target_branch[target_layer] = torch.nn.ELU(inplace=True)


class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x
    
    # define the function to resize the architecture kernel number
    def update_architecture(self, modification_num):
        update_times = int(modification_num + 1)
        for update_id in range(update_times):
            if torch.rand(1).item() < 0.05:
                self.prune_kernel()
            else:
                self.prune_inception()
        if torch.rand(1).item() < 0:
            if torch.rand(1).item() < 0.05:
                self.change_activation_function()
            else:
                self.change_inception_activation_function()
    
    def prune_kernel(self):
        target_branch = self.prelayer
        layer_choices = torch.tensor([0, 3, 6])
        target_layer = torch.randint(0, len(layer_choices), (1,)).item()
        target_layer = layer_choices[target_layer]
        
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

        if target_layer == 6:
            # if prone the last layer, we also need to modify a3
            for target_branch_id in range(1, 4):
                if target_branch_id == 1:
                    target_branch = self.a3.b1
                elif target_branch_id == 2:
                    target_branch = self.a3.b2
                elif target_branch_id == 3:
                    target_branch = self.a3.b3
                new_conv2 = nn.Conv2d(new_conv1_kernel_num, target_branch[0].out_channels, kernel_size=target_branch[0].kernel_size, bias=False)
                with torch.no_grad():
                    kept_indices = [i for i in range(target_branch[0].in_channels) if i != target_kernel]
                    new_conv2.weight.data = target_branch[0].weight.data[:, kept_indices, :, :]
                target_branch[0] = new_conv2
            target_branch = self.a3.b4
            new_conv2 = nn.Conv2d(new_conv1_kernel_num, target_branch[1].out_channels, kernel_size=target_branch[1].kernel_size, bias=False)
            with torch.no_grad():
                kept_indices = [i for i in range(target_branch[1].in_channels) if i != target_kernel]
                new_conv2.weight.data = target_branch[1].weight.data[:, kept_indices, :, :]
            target_branch[1] = new_conv2
        else:
            # else, decrement next conv input inside prelayer
            new_conv2 = nn.Conv2d(target_branch[target_layer + 3].in_channels - 1, target_branch[target_layer + 3].out_channels, kernel_size=target_branch[target_layer + 3].kernel_size, padding=1, bias=False)
            with torch.no_grad():
                kept_indices = [i for i in range(target_branch[target_layer + 3].in_channels) if i != target_kernel]
                new_conv2.weight.data = target_branch[target_layer + 3].weight.data[:, kept_indices, :, :]
            target_branch[target_layer + 3] = new_conv2
    
    def prune_inception(self):
        target_inception = torch.randint(0, 9, (1,)).item()
        if target_inception == 0:
            target_inception = self.a3
        elif target_inception == 1:
            target_inception = self.b3
        elif target_inception == 2:
            target_inception = self.a4
        elif target_inception == 3:
            target_inception = self.b4
        elif target_inception == 4:
            target_inception = self.c4
        elif target_inception == 5:
            target_inception = self.d4
        elif target_inception == 6:
            target_inception = self.e4
        elif target_inception == 7:
            target_inception = self.a5
        else:
            target_inception = self.b5
        ret, target_kernel = target_inception.prune_kernel()
        # if ret = none, no need to modify next layer's input
        if ret == None:
            return
        # compute coresponding target_kernel
        if ret == 1:
            target_kernel = target_kernel
        elif ret == 2:
            target_kernel = target_inception.branch1_out + target_kernel
        elif ret == 3:
            target_kernel = target_inception.branch1_out + target_inception.branch2_out + target_kernel
        else:
            target_kernel = target_inception.branch1_out + target_inception.branch2_out + target_inception.branch3_out + target_kernel
        # handle the last inception seperately as we need to modify FC layer
        if target_inception != self.b5:
            if target_inception == self.a3:
                target_inception_2 = self.b3
            elif target_inception == self.b3:
                target_inception_2 = self.a4
            elif target_inception == self.a4:
                target_inception_2 = self.b4
            elif target_inception == self.b4:
                target_inception_2 = self.c4
            elif target_inception == self.c4:
                target_inception_2 = self.d4
            elif target_inception == self.d4:
                target_inception_2 = self.e4
            elif target_inception == self.e4:
                target_inception_2 = self.a5
            else:
                target_inception_2 = self.b5
            
            for target_branch_id in range(1, 4):
                if target_branch_id == 1:
                    target_branch = target_inception_2.b1
                elif target_branch_id == 2:
                    target_branch = target_inception_2.b2
                elif target_branch_id == 3:
                    target_branch = target_inception_2.b3
                new_conv2 = nn.Conv2d(target_branch[0].in_channels - 1, target_branch[0].out_channels, kernel_size=target_branch[0].kernel_size, bias=False)
                with torch.no_grad():
                    kept_indices = [i for i in range(target_branch[0].in_channels) if i != target_kernel]
                    new_conv2.weight.data = target_branch[0].weight.data[:, kept_indices, :, :]
                target_branch[0] = new_conv2
            target_branch = target_inception_2.b4
            new_conv2 = nn.Conv2d(target_branch[1].in_channels - 1, target_branch[1].out_channels, kernel_size=target_branch[1].kernel_size, bias=False)
            with torch.no_grad():
                kept_indices = [i for i in range(target_branch[1].in_channels) if i != target_kernel]
                new_conv2.weight.data = target_branch[1].weight.data[:, kept_indices, :, :]
            target_branch[1] = new_conv2
        else:
            # update FC layer
            new_fc1_intput_features = self.linear.in_features - 1
            # update fc1
            new_fc1 = nn.Linear(new_fc1_intput_features, self.linear.out_features)
            output_length = 1 # gained by printing "output"
            # prune the neuron with least variance weights
            with torch.no_grad():
                start_index = target_kernel * output_length ** 2
                end_index = start_index + output_length ** 2
                new_fc1.weight.data = torch.cat([self.linear.weight.data[:, :start_index], self.linear.weight.data[:, end_index:]], dim=1)
                new_fc1.bias.data = self.linear.bias.data
            self.linear = new_fc1
    
    def change_activation_function(self):
        target_branch = self.prelayer
        layer_choices = torch.tensor([2, 5, 8])
        target_layer = torch.randint(0, len(layer_choices), (1,)).item()
        target_layer = layer_choices[target_layer]
        
        p1 = torch.rand(1).item()
        if p1 < 0.2:
            target_branch[target_layer] = torch.nn.ReLU(inplace=True)
        elif p1 < 0.4:
            target_branch[target_layer] = torch.nn.Tanh()
        elif p1 < 0.6:
            target_branch[target_layer] = torch.nn.LeakyReLU(inplace=True)
        elif p1 < 0.8:
            target_branch[target_layer] = torch.nn.Sigmoid()
        else:
            target_branch[target_layer] = torch.nn.ELU(inplace=True)

    def change_inception_activation_function(self):
        target_inception = torch.randint(0, 9, (1,)).item()
        if target_inception == 0:
            target_inception = self.a3
        elif target_inception == 1:
            target_inception = self.b3
        elif target_inception == 2:
            target_inception = self.a4
        elif target_inception == 3:
            target_inception = self.b4
        elif target_inception == 4:
            target_inception = self.c4
        elif target_inception == 5:
            target_inception = self.d4
        elif target_inception == 6:
            target_inception = self.e4
        elif target_inception == 7:
            target_inception = self.a5
        else:
            target_inception = self.b5
        target_inception.change_activation_function()
