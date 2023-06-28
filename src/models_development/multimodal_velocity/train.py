import torch
import torch.nn as nn
from tqdm import tqdm
# from tqdm.notebook import tqdm


def train(model: nn.Module,
          device: str,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion_classification: nn.Module,
          criterion_regression: nn.Module,
          bins_midpoints: torch.Tensor,
          epoch: int) -> tuple:
    """Train the model for one epoch

    Args:
        model (Model): The model to train
        device (string): The device to use (cpu or cuda)
        train_loader (Dataloader): The training data loader
        optimizer (Optimizer): The optimizer to use
        criterion_classification (Loss): The classification loss to use
        criterion_regression (Loss): The regression loss to use
        bins_midpoints (ndarray): The midpoints of the bins used to discretize
        the traversal costs
        epoch (int): The current epoch

    Returns:
        double, double, double: The training loss, the training accuracy and
        the training regression loss
    """
    # Initialize the training loss and accuracy
    train_loss = 0.
    train_correct = 0
    train_regression_loss = 0.
    
    # Configure the model for training
    # (good practice, only necessary if the model operates differently for
    # training and validation)
    model.train()
    
    # Add a progress bar
    train_loader_pbar = tqdm(train_loader, unit="batch")
    
    # Loop over the training batches
    for images,\
        traversal_costs,\
        traversability_labels,\
        linear_velocities in train_loader_pbar:
        
        # Print the epoch and training mode
        train_loader_pbar.set_description(f"Epoch {epoch} [train]")
        
        # Move images and traversal scores to GPU (if available)
        images = images.to(device)
        traversal_costs = traversal_costs.to(device)
        traversability_labels = traversability_labels.to(device)
        linear_velocities = linear_velocities.type(torch.float32).to(device)
        
        # Add a dimension to the linear velocities tensor
        linear_velocities.unsqueeze_(1)
        
        # Zero out gradients before each backpropagation pass, to avoid that
        # they accumulate
        optimizer.zero_grad()
        
        # Perform forward pass
        predicted_traversability_labels = model(images, linear_velocities)
        # predicted_traversal_scores = nn.Softmax(dim=1)(model(images))
        
        # Compute loss 
        loss = criterion_classification(predicted_traversability_labels, traversability_labels)
        
        # Print the batch loss next to the progress bar
        train_loader_pbar.set_postfix(batch_loss=loss.item())
        
        # Perform backpropagation (update weights)
        loss.backward()
        
        # Adjust parameters based on gradients
        optimizer.step()
        
        # Accumulate batch loss to average over the epoch
        train_loss += loss.item()
    
        # Get the number of correct predictions
        train_correct += torch.sum(
            torch.argmax(predicted_traversability_labels, dim=1) == traversability_labels
            ).item()
        
        # Compute the expected traversal cost over the bins
        expected_traversal_costs = torch.matmul(
            nn.Softmax(dim=1)(predicted_traversability_labels),
            bins_midpoints)
        
        # Compute and accumulate the batch loss
        train_regression_loss += criterion_regression(
            expected_traversal_costs[:, 0],
            traversal_costs).item()
    
    # Compute the losses and accuracies
    train_loss /= len(train_loader)
    train_accuracy = 100*train_correct/len(train_loader.dataset)
    train_regression_loss /= len(train_loader)
        
    return train_loss, train_accuracy, train_regression_loss
