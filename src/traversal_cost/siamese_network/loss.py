import torch
import torch.nn.functional as F


class SiameseLoss(torch.nn.Module):
    """Loss function for the siamese network

    Args:
        torch (nn.Module): Abstract class which represents a torch module
    """
    def __init__(self, margin: float=0.5):
        """Constructor of the class

        Args:
            margin (float, optional): Margin for the loss function
            Defaults to 2.0.
        """        
        super(SiameseLoss, self).__init__()
        
        # Introduce a margin to make sure the network does not
        # learn to output only 0
        self.margin = margin

    def forward(self,
                output1: torch.Tensor,
                output2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function

        Args:
            output1 (torch.Tensor): First output of the network
            output2 (torch.Tensor): Second output of the network

        Returns:
            torch.Tensor: Loss
        """
        # Compute the loss between the two outputs
        # (the relu function is equivalent to max(0, x))
        loss = F.relu(output1 - output2 + self.margin)
        
        # Take the mean over the batch
        loss = torch.mean(loss)
        
        return loss
