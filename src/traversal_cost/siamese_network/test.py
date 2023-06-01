import torch


def test(model,
         device,
         test_loader,
         criterion):
    """Test the model on the test set

    Args:
        model (Model): The model to test
        device (string): The device to use for the computations
        test_loader (Dataloader): The dataloader for the test set
        criterion (Loss): The loss function

    Returns:
        double: The average loss
    """
    # Testing
    test_loss = 0.

    # Configure the model for testing
    model.eval()

    with torch.no_grad():
        # Loop over the testing batches
        for features1, features2 in test_loader:
            
            features1 = features1.to(device)
            features2 = features2.to(device)
            
            # Perform forward pass
            predicted_costs1 = model(features1)
            predicted_costs2 = model(features2)

            # Compute loss
            loss = criterion(predicted_costs1, predicted_costs2)
            
            # Accumulate batch loss to average of the entire testing set
            test_loss += loss.item()


    # Compute the loss
    test_loss /= len(test_loader)
    
    return test_loss
