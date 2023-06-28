import torch
from torch.utils.data import Dataset
import torch.nn as nn

import numpy as np
import pandas as pd
import os

# Import custom packages
import params.supervised_learning


class SupervisedNetwork(nn.Module):
    
    def __init__(self, input_size: int):
        """Constructor of the class

        Args:
            input_size (int): The size of the input
        """
        super(SupervisedNetwork, self).__init__()
        
        # Setting up the Fully Connected Layers
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """

        output = self.mlp(x)
        
        return output
