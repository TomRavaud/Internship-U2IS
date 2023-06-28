import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset


class SupervisedLearningDataset(Dataset):
    """Custom Dataset class to represent our dataset
    It includes data and information about the data

    Args:
        Dataset (class): Abstract class which represents a dataset
    """
    
    def __init__(self,
                 traversal_costs_file: str,
                 features_directory:str,
                 transform: callable=None) -> None:
        """Constructor of the class

        Args:
            traversal_costs_file (string): Path to the csv file which contains
            feature of signals ids
            features_directory (string): Directory with all the features
            extracted from the signals
            transform (callable, optional): Optional transform to be applied
            on a sample. Defaults to None.
        """
        # Read the csv file
        self.data = pd.read_csv(traversal_costs_file,
                                converters={'id' : str})
        
        # Set the features data
        self.features_directory = features_directory
        
        # Set the transform
        self.transform = transform

    def __len__(self) -> int:
        """Return the size of the dataset

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Allow to access a sample by its index

        Args:
            idx (int): Index of a sample

        Returns:
            tuple: Sample at index idx
            (feature, true cost)
        """
        # Getting id with leading zeros
        id = self.data.loc[idx, "id"]
     
        # Get the signals ids corresponding to the pair at index idx
        features = np.load(
            os.path.join(
                self.features_directory, id + ".npy"))
        
        # Apply the transform
        if self.transform:
            features = self.transform(features)
            
        features = features.astype(np.float32)
        
        # Get the real cost according to the id 
        cost = self.data.loc[idx, "cost"].astype(np.float32)
        
        return features, cost
