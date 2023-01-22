"""
Fine-tuning is a way to perform transfer learning with a pre-trained
CNN architecture. The pre-trained model is further trained on the new task
using a small learning rate. This allows the model to adapt to the new task
while still leveraging the features learned from the original task.
"""

import torch
import torchvision.models as models

# Load the pre-trained AlexNet model
model = models.alexnet(pretrained=True)


model.classifier[-1] = torch.nn.Linear(4096, 1)

# Set the model to evaluation mode
model.eval()

# Input data
input = torch.randn(1, 3, 224, 224)

# Forward pass
output = model(input)