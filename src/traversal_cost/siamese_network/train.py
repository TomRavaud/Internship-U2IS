from tqdm import tqdm


def train(model,
          device,
          train_loader,
          optimizer,
          criterion,
          epoch):
    """Train the model for one epoch

    Args:
        model (Model): The model to train
        device (string): The device to use (cpu or cuda)
        train_loader (Dataloader): The training data loader
        optimizer (Optimizer): The optimizer to use
        criterion (Loss): The loss function to use
        epoch (int): The current epoch

    Returns:
        double : The training loss
    """
    # Initialize the training loss
    train_loss = 0.
    
    # Configure the model for training
    # (good practice, only necessary if the model operates differently for
    # training and validation)
    model.train()
    
    # Add a progress bar
    train_loader_pbar = tqdm(train_loader, unit="batch")
    
    # Loop over the training batches
    for features1, features2 in train_loader_pbar:
        
        # Print the epoch and training mode
        train_loader_pbar.set_description(f"Epoch {epoch} [train]")
        
        # Move features to GPU (if available)
        features1 = features1.to(device)
        features2 = features2.to(device)
        
        # Zero out gradients before each backpropagation pass, to avoid that
        # they accumulate
        optimizer.zero_grad()
        
        # Perform forward pass
        predicted_costs1 = model(features1)
        predicted_costs2 = model(features2)
        
        # Compute loss 
        loss = criterion(predicted_costs1,
                         predicted_costs2)
        
        # Print the batch loss next to the progress bar
        train_loader_pbar.set_postfix(batch_loss=loss.item())
        
        # Perform backpropagation (update weights)
        loss.backward()
        
        # Adjust parameters based on gradients
        optimizer.step()
        
        # Accumulate batch loss to average over the epoch
        train_loss += loss.item()
    
    
    # Compute the loss average over the epoch
    train_loss /= len(train_loader)
    
    return train_loss
