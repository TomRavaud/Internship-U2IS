from tqdm.notebook import tqdm
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
        for feature, real_cost in val_loader_pbar:

            # Print the epoch and validation mode
            val_loader_pbar.set_description(f"Epoch {epoch} [val]")

            # Move features to GPU (if available)
            feature = feature.to(device)
            real_cost = real_cost.to(device)
            
            # Perform forward pass (only, no backpropagation)
            predicted_cost = model(feature)

            # Compute loss
            loss = criterion(predicted_cost, real_cost)

            # Print the batch loss next to the progress bar
            val_loader_pbar.set_postfix(batch_loss=loss.item())

            # Accumulate batch loss to average over the epoch
            val_loss += loss.item()
            
        
    # Compute the loss
    val_loss /= len(val_loader)
    
    return val_loss
