"""
Step 3:

Design a NN model, train it with our training data, test its performance,
and optimize our hyperparameters to improve performance to a desired level
"""


# Import modules and libraries
import sys

# Add the location of the data_preparation directory at runtime
# (would be better to structure the files into packages)
sys.path.insert(
    0, "/home/tom/Traversability-Tom/Internship-U2IS/src/data_preparation")

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import torchvision

# TensorBoard for visualization
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

# Import custom module(s)
import data_preparation as dp


class LeNet5(nn.Module):
    """LeNet5 neural network implementation

    Args:
        nn.Module (class): The base class for all NN modules
    """
    def __init__(self):
        # Call the super() function to execute the parent nn.Module class's
        # __init__() method to initialize the class parameters
        super(LeNet5, self).__init__()
        
        # Instantiate two different convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Instantiate three different fully connected layers
        self.fc1 = nn.Linear(16*22*22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights for fully-connected and convolutional layers

        Args:
            module (nn.Module): Module to initialize weights
            (a layer is a module)
        """
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            # Xavier initialization from a uniform distribution
            nn.init.xavier_uniform_(module.weight)
        
    def forward(self, x):
        """Forward pass accross the NN

        Args:
            x (torch.*Tensor): Input data

        Returns:
            torch.*Tensor: Output prediction
        """
        # x shape: [batch size, image channels, image width, image height]
        # [2, 3, 100, 100]
        
        # Convolution and detector stages
        # In the original paper (LeCun 1998) the sigmoid function was used
        # but relu has proved to work better
        x = F.relu(self.conv1(x))
        
        # x shape: [batch size, kernel outputs,
        # (image width - kernel width + padding) \ stride + 1,
        # (image height - kernel height + padding) \ stride + 1]
        # [2, 6, 96, 96]
        
        # Pooling stage
        # In the original paper the average pooling was used but max pooling
        # has proved to work better
        x = F.max_pool2d(x, (2, 2))
        
        # x shape: [batch size, kernel outputs,
        # image width / kernel width if padding == kernel width,
        # image height / kernel height if padding == kernel height]
        # [2, 6, 48, 48]
        
        x = F.relu(self.conv2(x))
        
        # x shape: [batch size, kernel outputs,
        # (image width - kernel width + padding) \ stride + 1,
        # (image height - kernel height + padding) \ stride + 1]
        # [2, 16, 44, 44]
        x = F.max_pool2d(x, 2)
        
        # x shape:
        # [2, 16, 22, 22]
        
        # Flatten each example in the minibatch
        x = x.view(-1, int(x.nelement()/x.shape[0]))
        
        # x shape:
        # [1, 16*22*22 = 7744]
        
        x = F.relu(self.fc1(x))
        
        # x shape:
        # [2, 120]
        
        x = F.relu(self.fc2(x))
        
        # x shape:
        # [2, 84]
        
        x = self.fc3(x)
        
        # x shape:
        # [2, 1]
        
        return x


# Use a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Open TensorBoard
tensorboard = SummaryWriter()

# Create the model and move it to the GPU (if available)
model = LeNet5().to(device=device)
print(model)

# Display the architecture in TensorBoard
images, traversal_scores = next(iter(dp.train_loader))
images = images.to(device)
tensorboard.add_graph(model, images)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# An epoch is one complete pass of the training dataset through the network
NB_EPOCHS = 10

# Create a tensor to store the training and evaluation loss values 
loss_values = torch.empty(2, NB_EPOCHS)


# Loop over the epochs
for epoch in range(NB_EPOCHS):
    
    # Training
    train_loss = 0.
    
    # Configure the model for training
    # (good practice, only necessary if the model operates differently for
    # training and validation)
    model.train()
    
    # Loop over the training batches
    for images, traversal_scores in dp.train_loader:
        
        # Move images and traversal scores to GPU (if available)
        images = images.to(device)
        traversal_scores = traversal_scores.to(device)
        
        # Zero out gradients before each backpropagation pass, to avoid that
        # they accumulate
        optimizer.zero_grad()
        
        # Perform forward pass
        predicted_traversal_scores = model(images)
        
        # Compute loss 
        loss = criterion(predicted_traversal_scores, traversal_scores)
        
        # Perform backpropagation (compute gradients)
        loss.backward()
        
        # Adjust parameters based on gradients
        optimizer.step()
        
        # Accumulate batch loss to average over the epoch
        train_loss += loss.item()
    
    # Validation
    val_loss = 0.
    
    # Configure the model for testing
    model.eval()
    
    # Loop over the validation batches
    for images, traversal_scores in dp.val_loader:
        
        # Move images and traversal scores to GPU (if available)
        images = images.to(device)
        traversal_scores = traversal_scores.to(device)
        
        # Perform forward pass (only, no backpropagation)
        predicted_traversal_scores = model(images)
        
        # Compute loss
        loss = criterion(predicted_traversal_scores, traversal_scores)
        
        # Accumulate batch loss to average over the epoch
        val_loss += loss.item()
    
    
    # # Display the computed losses
    # print(f"Epoch {epoch}: Train Loss: {train_loss/len(dp.train_loader)}\
    #       Validation Loss: {val_loss/len(dp.val_loader)}")
    loss_values[0, epoch] = train_loss/len(dp.train_loader)
    loss_values[1, epoch] = val_loss/len(dp.val_loader)
    
    # Add the losses to TensorBoard
    tensorboard.add_scalar("train_loss", train_loss/len(dp.train_loader), epoch)
    tensorboard.add_scalar("val_loss", val_loss/len(dp.val_loader), epoch)
    

# Close TensorBoard
tensorboard.close()


# print(loss_values)

# plt.figure()

# plt.plot(range(NB_EPOCHS), loss_values[0], "b", label="train_loss")
# plt.plot(range(NB_EPOCHS), loss_values[1], "r--", label="val_loss")

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.show()
