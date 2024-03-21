import torch
import torch.nn as nn

class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = strides, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = nn.ReLU(self.bn1(self.conv1(x)))
        out = nn.ReLU(self.bn2(self.conv2(out)))
        return out
    
class deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        out = nn.ReLU(self.bn1(self.conv1(x)))
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.layer1_conv = double_conv2d_bn(1, 8)
        self.layer2_conv = double_conv2d_bn(8, 16)
        self.layer3_conv = double_conv2d_bn(16, 32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.deconv1 = deconv2d_bn(128, 64)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)
        self.layer9_conv = double_conv2d_bn(16, 8)
        self.layer10_conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        conv1 = self.layer1_conv(x)
        pool1 = self.max_pool2d(conv1)
        
        conv2 = self.layer2_conv(pool1)
        pool2 = self.max_pool2d(conv2)
        
        conv3 = self.layer3_conv(pool2)
        pool3 = self.max_pool2d(conv3)
        
        conv4 = self.layer4_conv(pool3)
        pool4 = self.max_pool2d(conv4)
        
        conv5 = self.layer5_conv(pool4)
        
        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1,conv4],dim=1)
        conv6 = self.layer6_conv(concat1)
        
        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.layer7_conv(concat2)
        
        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.layer8_conv(concat3)
        
        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,conv1],dim=1)
        conv9 = self.layer9_conv(concat4)
        output = self.layer10_conv(conv9)
        output = self.sigmoid(output)
        return output