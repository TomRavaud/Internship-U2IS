import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, nb_classes=10, nb_input_channels=3):
        super(ResNet50, self).__init__()

        self.nb_input_channels = nb_input_channels

        ## Multimodal images processing ##
        # Load the ResNet50 model with pretrained weights
        self.resnet50 = models.resnet50()
        
        # Replace the first convolutional layer to accept more than 3 channels
        self.resnet50.conv1 = nn.Conv2d(
            in_channels=nb_input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Replace the last fully-connected layer to have n classes as output
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features + 1, 128)

        # Add a second fully-connected layer
        self.fc = nn.Linear(128, nb_classes)

    def forward(self, x_img, x_dense):
        x = x_img

        # Forward pass through the ResNet50
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        # Concatenate the output of the ResNet50 with the dense input
        x = torch.cat((x, x_dense), dim=1)

        x = self.resnet50.fc(x)
        x = F.relu(x)
        x = self.fc(x)

        return x
