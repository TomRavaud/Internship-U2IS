import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import torch.nn as nn
import params.supervised_learning


class SupervisedNetwork(nn.Module):
    
    def __init__(self):
        super(SupervisedNetwork, self).__init__()
        
        # Setting up the Fully Connected Layers
        self.mlp = nn.Sequential(
            nn.Linear(params.supervised_learning.INPUT_FEATURE_SIZE, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
            )


    def forward(self, x):

        output = self.mlp(x)
        
        return output
        #returns the predicated cost according to the input vector
