import torch
import torch.nn as nn
from tqdm.notebook import tqdm


def validate(model: torch.nn.Module,
             device: str,
             val_loader: torch.utils.data.DataLoader,
             criterion_classification: nn.Module,
             criterion_regression: nn.Module,
             bins_midpoints: torch.Tensor,
             epoch: int) -> tuple:
    """Validate the model for one epoch

    Args:
        model (Model): The model to validate
        device (string): The device to use (cpu or cuda)
        val_loader (Dataloader): The validation data loader
        criterion_classification (Loss): The classification loss to use
        criterion_regression (Loss): The regression loss to use
        bins_midpoints (ndarray): The midpoints of the bins used to discretize
        the traversal costs
        epoch (int): The current epoch
        
    Returns:
        double, double, double: The validation loss, the validation accuracy
        and the validation regression loss
    """
    # Initialize the validation loss and accuracy
    val_loss = 0.
    val_correct = 0
    val_regression_loss = 0.
    
    # Configure the model for testing
    # (turn off dropout layers, batchnorm layers, etc)
    model.eval()
    
    # Add a progress bar
    val_loader_pbar = tqdm(val_loader, unit="batch")
    
    # Turn off gradients computation (the backward computational graph is
    # built during the forward pass and weights are updated during the backward
    # pass, here we avoid building the graph)
    with torch.no_grad():
        
        # Loop over the validation batches
        for images,\
            traversal_costs,\
            traversability_labels,\
            linear_velocities in val_loader_pbar:

            # Print the epoch and validation mode
            val_loader_pbar.set_description(f"Epoch {epoch} [val]")

            # Move images and traversal scores to GPU (if available)
            images = images.to(device)
            traversal_costs = traversal_costs.to(device)
            traversability_labels = traversability_labels.to(device)
            linear_velocities = linear_velocities.type(torch.float32).to(device)
            
            # Add a dimension to the linear velocities tensor
            linear_velocities.unsqueeze_(1)
            
            # Perform forward pass (only, no backpropagation)
            predicted_traversability_labels = model(images, linear_velocities)
            # predicted_traversal_scores = nn.Softmax(dim=1)(model(images))

            # Compute loss
            loss = criterion_classification(predicted_traversability_labels,
                                            traversability_labels)

            # Print the batch loss next to the progress bar
            val_loader_pbar.set_postfix(batch_loss=loss.item())

            # Accumulate batch loss to average over the epoch
            val_loss += loss.item()
            
            # Get the number of correct predictions
            val_correct += torch.sum(
                torch.argmax(predicted_traversability_labels, dim=1) == traversability_labels
                ).item()
            
            # Compute the expected traversal cost over the bins
            expected_traversal_costs = torch.matmul(
                nn.Softmax(dim=1)(predicted_traversability_labels),
                bins_midpoints)

            # Compute and accumulate the batch loss
            val_regression_loss += criterion_regression(
                expected_traversal_costs[:, 0],
                traversal_costs).item()
        
    # Compute the losses and accuracies
    val_loss /= len(val_loader)
    val_accuracy = 100*val_correct/len(val_loader.dataset)
    val_regression_loss /= len(val_loader)
    
    return val_loss, val_accuracy, val_regression_loss
