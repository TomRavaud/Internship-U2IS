import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Siamese Network class
    """
    def __init__(self, nb_input_features):
        """
        Constructor of the class
        """
        super(SiameseNetwork, self).__init__()
        
        self.fc = nn.Linear(nb_input_features, 1)
        
    
    def forward(self, x):
        
        x = F.relu(self.fc(x))
        
        return x
