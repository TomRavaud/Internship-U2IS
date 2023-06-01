import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):
    """Custom Dataset class to represent our dataset
    It includes data and information about the data

    Args:
        Dataset (class): Abstract class which represents a dataset
    """
    
    def __init__(self,
                 pairs_file,
                 features_directory):
        """Constructor of the class

        Args:
            pairs_file (string): Path to the csv file which contains
            pairs of signals ids
            features_directory (string): Directory with all the features
            extracted from the signals
        """
        # Read the csv file
        self.pairs_frame = pd.read_csv(pairs_file,
                                       converters={"id1": str,
                                                   "id2": str})
        
        # Initialize the name of the images directory
        self.features_directory = features_directory
        
    def __len__(self):
        """Return the size of the dataset

        Returns:
            int: Number of samples
        """
        return len(self.pairs_frame)

    def __getitem__(self, idx):
        """Allow to access a sample by its index

        Args:
            idx (int): Index of a sample

        Returns:
            list: Sample at index idx
            ([features from signal with id1,
              features from signal with id2])
        """
        # Get the signals ids corresponding to the pair at index idx
        id1 = self.pairs_frame.loc[idx, "id1"]
        id2 = self.pairs_frame.loc[idx, "id2"]
        
        # Get the features corresponding to the signals ids
        features1 = np.load(os.path.join(self.features_directory,
                                         id1 + ".npy")).astype(np.float32)
        features2 = np.load(os.path.join(self.features_directory,
                                         id2 + ".npy")).astype(np.float32)
        
        return features1, features2
