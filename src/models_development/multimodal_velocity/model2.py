"""
https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
"""

import torch.nn as nn
import torch
from torch import Tensor
from typing import Type


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None) -> None:
        
        super(BasicBlock, self).__init__()
        
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        
    def forward(self, x: Tensor) -> Tensor:
        
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            # If the output of the residual block has a different shape than
            # the input (stride > 1), then we need to downsample the input so
            # that the shapes match.
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return  out


class TraversabilityNet(nn.Module):
    
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock]=BasicBlock,
        num_classes: int=10,
        in_channels1: int=64,
        in_channels2: int=128,
        in_channels3: int=256,
        in_channels4: int=512,
        num_fc_features: int=256) -> None:
        
        super(TraversabilityNet, self).__init__()
        
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = in_channels1
        
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block,
                                       in_channels1,
                                       layers[0])
        self.layer2 = self._make_layer(block,
                                       in_channels2,
                                       layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       in_channels3,
                                       layers[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       in_channels4,
                                       layers[3],
                                       stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels4*self.expansion + 1, num_fc_features)
        self.fc2 = nn.Linear(num_fc_features, num_classes)
        
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1) -> nn.Sequential:
        
        downsample = None
        
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        
        layers = []
        
        # Add the first BasicBlock with "in_channels" being the number of 
        # output channels of the previous convolution
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                self.expansion,
                downsample
            )
        )
        
        # The number of input channels is constant for the rest of the blocks
        self.in_channels = out_channels * self.expansion
        
        # Add the remaining BasicBlocks
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor, x_dense: Tensor) -> Tensor:
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # # The spatial dimension of the final layer's feature 
        # # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Merge the image and numeric inputs
        x = torch.cat((x, x_dense), dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
