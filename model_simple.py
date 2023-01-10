import os
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

#selecting the device to train our model onto, i.e., CPU or a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")





#essai de modèle AlexNet sans régularisation

class AlexNet(nn.Module):

    def __init__(self,dropout: float = 0.4) -> None:
        super().__init__()
      
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #input 3*225*225 output 64 *61*61
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
            #input 64*61*61 output 64*30*30
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # target output size of 6*6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            #nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 1028),
            #nn.BatchNorm1d(1028),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1028, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(128,32),
            nn.BatchNorm1d(32),
            
            #m = nn.LeakyReLU(0.1)
            
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(p=dropout),
            
            
            nn.Linear(32,1),

            nn.Sigmoid()
            
           
        )

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
           torch.nn.init.xavier_uniform(m.weight)
           m.bias.data.fill_(0.)
        elif isinstance(m, nn.Conv2d):
           torch.nn.init.xavier_uniform(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        #print(x)
       
        #x = self.avgpool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)#same as x.view
        #print(x.shape)
        x = self.classifier(x)
        return x
        
        
                

        
#######
#######
#######




class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.Leakyrelu = nn.LeakyReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.Leakyrelu(self.batch_norm1(self.conv1(x)))
        
        x = self.Leakyrelu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.Leakyrelu(x)
        
        return x
        
        
        
### pour representer un simple block

#pas de changement de channel
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, dim_changed=None, stride=1):
        super(ResBlock, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.dim_changed = dim_changed
        self.stride = stride
        self.Leakyrelu = nn.LeakyReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.Leakyrelu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
          
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.Leakyrelu(x)
      return x

######
        
 ######c'est pour relier les blocs et les nommbres de channels suivenet ce qui est écrit sur research papers
 
 #####
 
 
      
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.Leakyrelu = nn.LeakyReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion,num_classes)
        #self.fc1 = nn.Linear(128, 32)
        #self.fc2 = nn.Linear(32, num_classes)
        
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
           torch.nn.init.xavier_uniform(m.weight)
           m.bias.data.fill_(0.)
        elif isinstance(m, nn.Conv2d):
           torch.nn.init.xavier_uniform(m.weight)

        
    def forward(self, x):
        x = self.Leakyrelu(self.batch_norm1(self.conv1(x)))
        
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
       # x = self.fc1(x)
    #x = self.fc2(x)
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        #if the block is bottleneck whe have to change the dimensions
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(1,blocks):#because the first one is already added
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
        
        
        
        
        
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)



        
    
