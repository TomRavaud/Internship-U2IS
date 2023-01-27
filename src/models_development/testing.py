# Import libraries
import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models
from torchvision import transforms

# TensorBoard for visualization
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import matplotlib.pyplot as plt

# Import modules and libraries
import sys

# Add the location of the data_preparation directory at runtime
# (would be better to structure the files into packages)
sys.path.insert(
    0, "/home/tom/Traversability-Tom/Internship-U2IS/src/data_preparation")

# Import custom module(s)
import data_preparation as dp


#TODO: why is not the GPU available?
# Use a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# Load the pre-trained AlexNet model
model = models.resnet18().to(device=device)

# Replace the last layer by a fully-connected one with 1 output
model.fc = nn.Linear(model.fc.in_features, 1)

model.load_state_dict(torch.load("resnet18_fine_tuned.params"))

model.eval()

images, traversal_scores = next(iter(dp.test_loader))

images = images.to(device)
traversal_scores = traversal_scores.to(device)

# Perform forward pass (only, no backpropagation)
predicted_traversal_scores = model(images)

print(predicted_traversal_scores)
print(traversal_scores)

for image in images:
    plt.imshow(transforms.ToPILImage()(image), cmap="gray")
    plt.show()