import torch


def test_supervised(model,
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
        for feature, real_cost in test_loader:
            
            feature = feature.to(device)
            real_cost = real_cost.to(device)
            
            # Perform forward pass
            predicted_cost = model(feature)
            print(f"predicted_cost is {predicted_cost}")
            print(f"real cost is {real_cost}")

            # Compute loss
            loss = criterion(predicted_cost, real_cost)
            
            # Accumulate batch loss to average of the entire testing set
            test_loss += loss.item()


    # Compute the loss
    test_loss /= len(test_loader)
    
    return test_loss
