from tqdm import tqdm
import torch


def validate(model,
             device,
             val_loader,
             criterion,
             epoch):
    """Validate the model for one epoch

    Args:
        model (Model): The model to validate
        device (string): The device to use (cpu or cuda)
        val_loader (Dataloader): The validation data loader
        criterion (Loss): The loss function to use
        epoch (int): The current epoch
        
    Returns:
        double: The validation loss
    """
    # Initialize the validation loss
    val_loss = 0.
    
    # Configure the model for testing
    # (turn off dropout layers, batchnorm layers, etc)
    model.eval()
    
    # Add a progress bar
    val_loader_pbar = tqdm(val_loader, unit="batch")
    
    # Turn off gradients computation (the backward computational graph is built during
    # the forward pass and weights are updated during the backward pass, here we avoid
    # building the graph)
    with torch.no_grad():
        # Loop over the validation batches
        for features1, features2 in val_loader_pbar:

            # Print the epoch and validation mode
            val_loader_pbar.set_description(f"Epoch {epoch} [val]")

            # Move features to GPU (if available)
            features1 = features1.to(device)
            features2 = features2.to(device)
            
            # Perform forward pass (only, no backpropagation)
            predicted_costs1 = model(features1)
            predicted_costs2 = model(features2)

            # Compute loss
            loss = criterion(predicted_costs1, predicted_costs2)

            # Print the batch loss next to the progress bar
            val_loader_pbar.set_postfix(batch_loss=loss.item())

            # Accumulate batch loss to average over the epoch
            val_loss += loss.item()
            
        
    # Compute the loss
    val_loss /= len(val_loader)
    
    return val_loss
