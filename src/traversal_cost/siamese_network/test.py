import torch


def test(model: torch.nn.Module,
         device: str,
         test_loader: torch.utils.data.DataLoader,
         criterion: torch.nn.Module) -> tuple:
    """Test the model on the test set

    Args:
        model (Model): The model to test
        device (string): The device to use for the computations
        test_loader (Dataloader): The dataloader for the test set
        criterion (Loss): The loss function

    Returns:
        float, float: The average loss, the accuracy
    """
    # Testing
    test_loss = 0.
    
    # Initialize the number of correct ranking predictions in order to compute
    # the accuracy
    test_correct = 0

    # Configure the model for testing
    model.eval()

    with torch.no_grad():
        # Loop over the testing batches
        for features1, features2, id1, id2 in test_loader:
            
            # Move features to GPU (if available)
            features1 = features1.to(device)
            features2 = features2.to(device)
            
            # Perform forward pass
            predicted_costs1 = model(features1)
            predicted_costs2 = model(features2)

            # Compute loss
            loss = criterion(predicted_costs1, predicted_costs2)
            
            # Accumulate batch loss to average of the entire testing set
            test_loss += loss.item()
            
            # Get the number of correct predictions
            test_correct += torch.sum(
                predicted_costs1 < predicted_costs2).item()
            
            # NOTE: This is just for debugging purposes
            # indx = predicted_costs1 >= predicted_costs2
            # for i, elt in enumerate(indx):
            #     if elt:
            #         print(id1[i], id2[i])

    # Compute the loss
    test_loss /= len(test_loader)
    
    # Compute the accuracy
    test_accuracy = 100*test_correct/len(test_loader.dataset)
    
    return test_loss, test_accuracy
