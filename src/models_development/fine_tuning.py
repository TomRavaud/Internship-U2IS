"""
Fine-tuning is a way to perform transfer learning with a pre-trained
CNN architecture. The pre-trained model is further trained on the new task
using a small learning rate. This allows the model to adapt to the new task
while still leveraging the features learned from the original task.
"""

# Import libraries
import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models

# A module to print a model summary (outputs shape, number of parameters, ...)
import torchsummary

# TensorBoard for visualization
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

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

# Open TensorBoard
tensorboard = SummaryWriter()

# Load the pre-trained AlexNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device=device)
# model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# Replace the last layer by a fully-connected one with 1 output
# model.classifier[-1] = nn.Linear(4096, 1)  # AlexNet
model.fc = nn.Linear(model.fc.in_features, 1)

# Display the architecture in TensorBoard
images, traversal_scores = next(iter(dp.train_loader))
images = images.to(device)
tensorboard.add_graph(model, images)


# print(model)
# print(torchsummary.summary(model, (3, 100, 100)))


# Initialize the last layer using Xavier initialization
nn.init.xavier_uniform_(model.fc.weight)


# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# An epoch is one complete pass of the training dataset through the network
NB_EPOCHS = 10

# Loop over the epochs
for epoch in range(NB_EPOCHS):
    
    # Training
    train_loss = 0.
    
    # Configure the model for training
    # (good practice, only necessary if the model operates differently for
    # training and validation)
    model.train()
    
    # Add a progress bar
    train_loader_pbar = tqdm(dp.train_loader, unit="batch")
    
    # Loop over the training batches
    for images, traversal_scores in train_loader_pbar:
        
        # Print the epoch and training mode
        train_loader_pbar.set_description(f"Epoch {epoch} [train]")
        
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
        
        # Print the batch loss next to the progress bar
        train_loader_pbar.set_postfix(batch_loss=loss.item())
        
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
    
    # Add a progress bar
    val_loader_pbar = tqdm(dp.val_loader, unit="batch")
    
    # Loop over the validation batches
    for images, traversal_scores in val_loader_pbar:
        
        # Print the epoch and validation mode
        val_loader_pbar.set_description(f"Epoch {epoch} [val]")
        
        # Move images and traversal scores to GPU (if available)
        images = images.to(device)
        traversal_scores = traversal_scores.to(device)
        
        # Perform forward pass (only, no backpropagation)
        predicted_traversal_scores = model(images)
        
        # Compute loss
        loss = criterion(predicted_traversal_scores, traversal_scores)
        
        # Print the batch loss next to the progress bar
        val_loader_pbar.set_postfix(batch_loss=loss.item())
        
        # Accumulate batch loss to average over the epoch
        val_loss += loss.item()
    
    
    # # Display the computed losses
    # print(f"Epoch {epoch}: Train Loss: {train_loss/len(dp.train_loader)}\
    #       Validation Loss: {val_loss/len(dp.val_loader)}")
    # loss_values[0, epoch] = train_loss/len(dp.train_loader)
    # loss_values[1, epoch] = val_loss/len(dp.val_loader)
    
    # Add the losses to TensorBoard
    tensorboard.add_scalar("train_loss", train_loss/len(dp.train_loader), epoch)
    tensorboard.add_scalar("val_loss", val_loss/len(dp.val_loader), epoch)


# Testing
test_loss = 0.

# Loop over the testing batches
for images, traversal_scores in dp.test_loader:
    
    images = images.to(device)
    traversal_scores = traversal_scores.to(device)
    
    # Perform forward pass
    predicted_traversal_scores = model(images)
    
    # Compute loss
    loss = criterion(predicted_traversal_scores, traversal_scores)
    
    # Accumulate batch loss to average of the entire testing set
    test_loss += loss.item()

print(f"Test loss: {test_loss}")

# Close TensorBoard
tensorboard.close()

# Save the model parameters
torch.save(model.state_dict(), "resnet18_fine_tuned.params")
