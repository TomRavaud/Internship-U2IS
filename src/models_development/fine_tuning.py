"""
Fine-tuning is a way to perform transfer learning with a pre-trained
CNN architecture. The pre-trained model is further trained on the new task
using a small learning rate. This allows the model to adapt to the new task
while still leveraging the features learned from the original task.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# A module to print a model summary (outputs shape, number of parameters, ...)
import torchsummary


# Load the pre-trained AlexNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# Replace the last layer by a fully-connected one
# model.classifier[-1] = nn.Linear(4096, 1)  # AlexNet
model.fc = nn.Linear(model.fc.in_features, 1)

# Initialize the last layer using Xavier initialization
nn.init.xavier_uniform_(model.fc.weight)


# print(model)
print(torchsummary.summary(model, (3, 224, 224)))

# Set the model to evaluation mode
model.eval()

# Input data
input = torch.randn(1, 3, 224, 224)

# Forward pass
output = model(input)

# print(output)
