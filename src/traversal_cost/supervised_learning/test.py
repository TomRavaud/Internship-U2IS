import torch


def test(model: torch.nn.Module,
         device: str,
         test_loader: torch.utils.data.DataLoader,
         criterion: torch.nn.Module) -> float:
    """Test the model on the test set

    Args:
        model (torch.nn.Module): The model to test
        device (str): The device to use for the computations
        test_loader (torch.utils.data.DataLoader): The dataloader for the
        test set
        criterion (torch.nn.Module): The loss function

    Returns:
        double: The average loss
    """
    # Testing
    test_loss = 0.

    # Configure the model for testing
    model.eval()

    with torch.no_grad():
        
        # Loop over the testing batches
        for features, true_costs in test_loader:
            
            # Move features and costs to GPU (if available)
            features = features.to(device)
            true_costs = true_costs.to(device)
            
            # Add a dimension to the linear velocities tensor
            true_costs.unsqueeze_(1)
            
            # Perform forward pass
            predicted_costs = model(features)

            # Compute loss
            loss = criterion(predicted_costs, true_costs)
            
            # Accumulate batch loss to average of the entire testing set
            test_loss += loss.item()


    # Compute the loss
    test_loss /= len(test_loader)
    
    return test_loss
