import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = strides, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = strides, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.double_conv(x)
    
    def prune_kernel(self):
        target_branch = self.double_conv
        # both layer have equal probability to be pruned as they both do not modifiy feature map's dimension
        layer_choices = torch.tensor([0, 3])
        target_layer = torch.randint(0, len(layer_choices), (1,)).item()
        target_layer = layer_choices[target_layer]

        # prune target branch
        new_conv1_kernel_num = target_branch[target_layer].out_channels - 1
        if new_conv1_kernel_num == 0:
            return None
        new_conv1 = nn.Conv2d(target_branch[target_layer].in_channels, new_conv1_kernel_num, kernel_size=target_branch[target_layer].kernel_size, stride = target_branch[target_layer].stride, padding = target_branch[target_layer].padding, bias=False)
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
        if target_layer == 3:
            # if prune last conv layer, we need to decrement next layer's input
            return target_kernel
        else:
            # else, we need to decrement next conv layer's input inside double_conv2d_bn
            new_conv2 = nn.Conv2d(target_branch[target_layer + 3].in_channels - 1, target_branch[target_layer + 3].out_channels, kernel_size=target_branch[target_layer + 3].kernel_size, stride = target_branch[target_layer + 3].stride, padding = target_branch[target_layer + 3].padding, bias=False)
            with torch.no_grad():
                kept_indices = [i for i in range(target_branch[target_layer + 3].in_channels) if i != target_kernel]
                new_conv2.weight.data = target_branch[target_layer + 3].weight.data[:, kept_indices, :, :]
            target_branch[target_layer + 3] = new_conv2
            return None
        
    def decre_input(self, target_kernel, is_crossing = False, input_offset = 0):
        target_branch = self.double_conv
        if is_crossing == True:
            # for the feature when we need to match exact kernel for encoder, which accepts two output of decoder
            target_kernel += (target_branch[0].in_channels // 2) * input_offset
        new_conv1 = nn.Conv2d(target_branch[0].in_channels - 1, target_branch[0].out_channels, kernel_size=target_branch[0].kernel_size, stride = target_branch[0].stride, padding = target_branch[0].padding, bias=False)
        with torch.no_grad():
            kept_indices = [i for i in range(target_branch[0].in_channels) if i != target_kernel]
            new_conv1.weight.data = target_branch[0].weight.data[:, kept_indices, :, :]
        target_branch[0] = new_conv1

    
class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv2d_bn(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)
    
    def prune_kernel(self):
        target_sequential = self.maxpool_conv[1]
        target_kernel = target_sequential.prune_kernel()
        return target_kernel
    
    def decre_input(self, target_kernel, is_crossing = False, input_offset = 0):  
        target_branch = self.maxpool_conv[1]
        target_branch.decre_input(target_kernel, is_crossing, input_offset)

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = double_conv2d_bn(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # the error of feature map comes from the Carvana provided dataset is 1918 * 1280, where 1918 will round up when divided by 2 multiple times
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
    def prune_kernel(self):
        # has lower probability to prune up as it changes feature map dimension, which is more sensitive
        if torch.rand(1).item() < 1:
            target_sequential = self.up
            new_conv1_kernel_num = target_sequential.out_channels - 1
            if new_conv1_kernel_num <= 1:
                return None
            new_convtransposed1 = nn.ConvTranspose2d(target_sequential.in_channels, new_conv1_kernel_num, kernel_size=target_sequential.kernel_size, stride = target_sequential.stride)
            weight_variances = torch.var(target_sequential.weight.data, dim = [0, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            with torch.no_grad():
                # NOTICE! For nn.Conv2d, its weight tensor is [Out Channel, In Channel, Kernel Height, Kernel Weight], For nn.ConvTranspose2d, its weight tensor is [In Channel, OutChannel, Kernel Height, Kernel Weight]
                new_convtransposed1.weight.data = torch.cat([target_sequential.weight.data[:, :target_kernel], target_sequential.weight.data[:, target_kernel+1:]], dim=1)
                new_convtransposed1.bias.data = torch.cat([target_sequential.bias.data[:target_kernel], target_sequential.bias.data[target_kernel+1:]])
            self.up = new_convtransposed1

            # then decrement next sequential's input
            self.conv.decre_input(target_kernel, is_crossing=False, input_offset=0)
            return None
        else:
            target_sequential = self.conv
            target_kernel = target_sequential.prune_kernel()
            return target_kernel
    
    def decre_input(self, target_kernel):
        # only decrement self.up
        target_sequential = self.up
        new_convtransposed1 = nn.ConvTranspose2d(target_sequential.in_channels - 1, target_sequential.out_channels, kernel_size=target_sequential.kernel_size, stride=target_sequential.stride)
        with torch.no_grad():
            kept_indices = [i for i in range(target_sequential.in_channels) if i != target_kernel]
            # NOTICE! For nn.Conv2d, its weight tensor is [Out Channel, In Channel, Kernel Height, Kernel Weight], For nn.ConvTranspose2d, its weight tensor is [In Channel, OutChannel, Kernel Height, Kernel Weight]
            new_convtransposed1.weight.data = target_sequential.weight.data[kept_indices, :, :, :]
            new_convtransposed1.bias.data = target_sequential.bias.data
        self.up = new_convtransposed1
    
class UNet(nn.Module):
    def __init__(self, in_channels = 3, num_class = 2):
        super(UNet,self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class

        self.preprocess = double_conv2d_bn(in_channels, 64)
        self.down1 = Encoder(64, 128)
        self.down2 = Encoder(128, 256)
        self.down3 = Encoder(256, 512)
        self.down4 = Encoder(512, 1024)
        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)
        self.postprocess = nn.Conv2d(64, num_class, kernel_size=1)
        
    def forward(self,x):
        x1 = self.preprocess(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        '''print(self.up1.up.weight.shape)
        print(self.up2.up.weight.shape)
        print(self.up3.up.weight.shape)
        print(self.up4.up.weight.shape)
        print(self.up1.conv.double_conv[0].weight.shape)
        print(self.up2.conv.double_conv[0].weight.shape)
        print(self.up3.conv.double_conv[0].weight.shape)
        print(self.up4.conv.double_conv[0].weight.shape)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(x4.shape)
        print(x5.shape)'''
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.postprocess(x)
        return output
    
    def update_architecture(self, modification_num):
        update_times = int(modification_num + 1)
        for update_id in range(update_times):
            # low probability to prune preprocess as it are sensitive, post process can not be pruned as its output channel is fixed
            if torch.rand(1).item() < 0.005:
                self.prune_preprocess()
            else:
                self.prune_encoder_decoder()
    
    def prune_preprocess(self):
        target_sequential = self.preprocess
        target_kernel = target_sequential.prune_kernel()
        if target_kernel != None:
            self.down1.decre_input(target_kernel, is_crossing=False, input_offset=0)
            self.up4.conv.decre_input(target_kernel, is_crossing=True, input_offset=1)

    def prune_encoder_decoder(self):
        # equal probability to prune encoder and decoder
        if torch.rand(1).item() < 0.5:
            # prune encoder
            encoder_choices = torch.tensor([1, 2, 3, 4])
            target_encoder = torch.randint(0, len(encoder_choices), (1,)).item()
            target_encoder = encoder_choices[target_encoder]
            if target_encoder == 1:
                target_encoder = self.down1
                next_encoder = self.down2
                corresponding_decoder = self.up3

                target_kernel = target_encoder.prune_kernel()
                if target_kernel != None:
                    next_encoder.decre_input(target_kernel, is_crossing=False, input_offset=0)
                    corresponding_decoder.conv.decre_input(target_kernel, is_crossing=True, input_offset=1)
            elif target_encoder == 2:
                target_encoder = self.down2
                next_encoder = self.down3
                corresponding_decoder = self.up2

                target_kernel = target_encoder.prune_kernel()
                if target_kernel != None:
                    next_encoder.decre_input(target_kernel, is_crossing=False, input_offset=0)
                    corresponding_decoder.conv.decre_input(target_kernel, is_crossing=True, input_offset=1)
            elif target_encoder == 3:
                target_encoder = self.down3
                next_encoder = self.down4
                corresponding_decoder = self.up1

                target_kernel = target_encoder.prune_kernel()
                if target_kernel != None:
                    next_encoder.decre_input(target_kernel, is_crossing=False, input_offset=0)
                    corresponding_decoder.conv.decre_input(target_kernel, is_crossing=True, input_offset=1)
            else:
                target_encoder = self.down4
                corresponding_decoder = self.up1
                target_kernel = target_encoder.prune_kernel()
                if target_kernel != None:
                    corresponding_decoder.decre_input(target_kernel)
        else:
            # prune decoder
            decoder_choices = torch.tensor([1, 2, 3, 4])
            target_decoder = torch.randint(0, len(decoder_choices), (1,)).item()
            target_decoder = decoder_choices[target_decoder]
            if target_decoder == 1:
                target_decoder = self.up1
                next_decoder = self.up2
                target_kernel = target_decoder.prune_kernel()
                if target_kernel != None:
                    next_decoder.decre_input(target_kernel)
            elif target_decoder == 2:
                target_decoder = self.up2
                next_decoder = self.up3
                target_kernel = target_decoder.prune_kernel()
                if target_kernel != None:
                    next_decoder.decre_input(target_kernel)
            elif target_decoder == 3:
                target_decoder = self.up3
                next_decoder = self.up4
                target_kernel = target_decoder.prune_kernel()
                if target_kernel != None:
                    next_decoder.decre_input(target_kernel)
            else:
                target_decoder = self.up4
                target_kernel = target_decoder.prune_kernel()
                if target_kernel != None:
                    # need to decrement postprocess input
                    new_conv2 = nn.Conv2d(self.postprocess.in_channels - 1, self.postprocess.out_channels, kernel_size=self.postprocess.kernel_size)
                    with torch.no_grad():
                        kept_indices = [i for i in range(self.postprocess.in_channels) if i != target_kernel]
                        new_conv2.weight.data = self.postprocess.weight.data[:, kept_indices, :, :]
                        new_conv2.bias.data = self.postprocess.bias.data
                    self.postprocess = new_conv2
    