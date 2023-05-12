# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureExtractor(nn.Module):
    def __init__(self,cnn,feature_layer = 19):
        super(FeatureExtractor,self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self,x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self,in_channels = 64,k = 3,n = 64,s = 1):
        super(residualBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,n,k,stride = s,padding = 1)
        self.bn1 = nn.BatchNorm2d(n)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(n,n,k,stride = s,padding = 1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self,x):
        y = self.prelu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self,inchannel,outchannel,size):
        super(upsampleBlock,self).__init__()
        self.conv1 = nn.Conv2d(inchannel,outchannel,3,1,1)
        self.upsample = nn.Upsample(size = [size,size])
        self.prelu = nn.PReLU()

    def forward(self,x):
        x = self.prelu(self.upsample(self.conv1(x)))
        return x


class Generator(nn.Module):
    def __init__(self,n_residual_blocks,high_resolution,scale_factor):
        super(Generator,self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.high_resolution = high_resolution
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(1,64,9,stride = 1,padding = 4)
        self.prelu = nn.PReLU()
        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1),residualBlock())

        self.conv2 = nn.Conv2d(64,64,3,stride = 1,padding = 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.low_resolution = self.high_resolution / self.scale_factor
        for i in range(1,int(math.log2(self.scale_factor)) + 1):
            size = int(max(self.high_resolution * (2 ** i) / scale_factor,self.low_resolution * (2 ** i)))
            self.add_module('upsample' + str(i),upsampleBlock(inchannel = 64,outchannel = 64,size = size))

        self.conv3 = nn.Conv2d(64,1,9,stride = 1,padding = 4)

    def forward(self,x):
        x = self.prelu(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(1,int(math.log2(self.scale_factor)) + 1):
            x = self.__getattr__('upsample' + str(i))(x)

        return self.conv3(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,stride = 1,padding = 1)

        self.conv2 = nn.Conv2d(64,64,3,stride = 2,padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,stride = 1,padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,3,stride = 2,padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256,3,stride = 1,padding = 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,256,3,stride = 2,padding = 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256,512,3,stride = 1,padding = 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512,512,3,stride = 2,padding = 1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512,1,1,stride = 1,padding = 1)

        self.lrelu=nn.LeakyReLU()
    def forward(self,x):
        x = self.lrelu(self.conv1(x))

        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        x = self.lrelu(self.bn6(self.conv6(x)))
        x = self.lrelu(self.bn7(self.conv7(x)))
        x = self.lrelu(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x,x.size()[2:])).view(x.size()[0],-1)
