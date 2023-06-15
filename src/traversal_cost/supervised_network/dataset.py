import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset


class SupervisedNetworkDataset(Dataset):
    """Custom Dataset class to represent our dataset
    It includes data and information about the data

    Args:
        Dataset (class): Abstract class which represents a dataset
    """
    
    def __init__(self, csv_file_root, features_directory):
        """Constructor of the class

        Args:
            csv file (string): Path to the csv file which contains
            feature of signals ids
            features_directory (string): Directory with all the features
            extracted from the signals
        """
        # Read the csv file
        self.data = pd.read_csv(csv_file_root, converters={'id' : str})
        
        #Set the features data
        self.features_directory = features_directory

    def __len__(self):
        """Return the size of the dataset

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Allow to access a sample by its index

        Args:
            idx (int): Index of a sample

        Returns:
            list: Sample at index idx
            (feature, real cost)
        """
        #Getting id with leading zeros
        id = self.data.loc[idx, "id"]
     
        # Get the signals ids corresponding to the pair at index idx
        feature = np.load(os.path.join(self.features_directory, id + ".npy")).astype(np.float32)        
        
        # Get the real cost according to the id 
        cost = self.data.loc[idx, "cost"].astype(np.float32)
        
        return feature, cost
