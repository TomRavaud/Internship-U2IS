import torch
import torch.nn.functional as F

class SupervisedLoss(torch.nn.Module):
    """Loss function for the siamese network

    Args:
        torch (nn.Module): Abstract class which represents a torch module
    """
    def __init__(self):
        
        super(SupervisedLoss, self).__init__()
        

    def forward(self, estimated_cost, real_cost):
       
        # Compute the loss between the two outputs : estimated cost and real cost
        # Using the MSE loss function
        
        loss = F.mse_loss(estimated_cost, real_cost)
        
        return loss
