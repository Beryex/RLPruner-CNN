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
        # need to store the value of input channels from corresponding encoder
        self.encoder_channels = in_channels // 2
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # the error of feature map comes from the Carvana provided dataset is 1918 * 1280, where 1918 will round up when divided by 2 multiple times
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_class = 2):
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
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.postprocess(x)
        return output
    