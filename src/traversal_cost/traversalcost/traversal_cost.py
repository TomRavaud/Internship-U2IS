import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import torch.nn as nn
import torch


# Import custom packages
import params.traversal_cost


def dimensionality_reduction(dataset: str,
                             method: str="pca",
                             nb_components: int=2) -> tuple:
    """Apply dimensionality reduction to the features

    Args:
        dataset (str): The path to the dataset
        method (str, optional): The method to use for dimensionality reduction.
        Defaults to "pca".
        nb_components (int, optional): The number of components to keep.
        Defaults to 2.

    Returns:
        tuple: (dimensionality reduction function, scaler)
    """    
    
    # Set the path to the features directory
    features_directory = dataset + "features/"
    
    # Initialize an empty list to store the loaded arrays
    features_all = []
    
    # Create a list to store the ids of the features
    features_ids = []
    
    # Iterate over the files in the directory
    for features_file in os.listdir(features_directory):
        
        # Append the id to the list
        features_ids.append(features_file.split(".")[0])
        
        # Load the features
        features = np.load(features_directory + features_file)
        
        # Append the features to the list
        features_all.append(features[None, :])
        
    # Concatenate the features
    features_all = np.concatenate(features_all, axis=0)
    
    # Scale the dataset
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_all)

    # Apply PCA
    if method == "pca":
        
        # Compute the PCA and get the first two principal components
        technique = PCA(n_components=nb_components)
        features_reduced = technique.fit_transform(features_scaled)

        # Display the coefficients of the first principal components
        plt.matshow(technique.components_, cmap="viridis")
        plt.colorbar()
        # plt.xticks(range(3),
        #            [
        #                "var [roll]",
        #                "var [pitch]",
        #                "var [z acc]",
        #             ],
        #            rotation=60,
        #            ha="left")
        plt.yticks(range(nb_components),
                   ["Principal component " +
                    str(i+1) for i in range(nb_components)])
        plt.xlabel("Feature")

    # Apply t-SNE
    elif method == "tsne":
        
        # Compute the t-SNE
        technique = TSNE(random_state=42, n_components=nb_components)
        features_reduced = technique.fit_transform(features_scaled)
    
    # Visualize the data if the number of components is less or equal to 2
    if nb_components <= 2:
        
        # Read the csv file containing the labels
        labels_df = pd.read_csv(dataset + "labels.csv",
                                converters={"id": str})
        
        # Convert the list of ids to a numpy array
        features_ids = np.array(features_ids)[:, None]
        
        # Concatenate the ids and the first components
        features_data = np.concatenate((features_ids,
                                        features_reduced),
                                       axis=1)
        
        components_str = ["component" + str(i+1) for i in range(nb_components)]
        
        # Create a dataframe from the features
        features_df = pd.DataFrame(
            features_data,
            columns=["id"] + components_str)
        
        # Convert the first components columns to float
        for component in components_str:
            features_df[component] = features_df[component].astype(float)
        
        # Merge the features and the labels on the id
        merged_df = pd.merge(features_df, labels_df, on="id")
        
        # Create a figure
        plt.figure()

        # Terrain classes
        plt.subplot(1, 2, 1)
        
        # Get the unique labels
        labels_unique = set(list(merged_df["terrain_class"]))
        
        # Iterate over the labels
        for label in labels_unique:
            
            # Get the samples of the current terrain class
            df = merged_df[merged_df["terrain_class"] == label]
            
            plt.scatter(df["component1"],
                        df["component2"] if nb_components == 2 else
                        [0]*len(df["component1"]),
                        label=label,
                        c=params.traversal_cost.colors[label],
                        )

        plt.legend()
        plt.xlabel("Component 1")
        
        if nb_components == 2:
            plt.ylabel("Component 2")

        # Velocities
        plt.subplot(1, 2, 2)
        
        # Get the unique velocities
        velocities_unique = set(list(merged_df["linear_velocity"]))
        
        # Iterate over the velocities
        for velocity in sorted(velocities_unique):
            
            # Get the samples of the current velocity
            df = merged_df[merged_df["linear_velocity"] == velocity]

            plt.scatter(df["component1"],
                        df["component2"] if nb_components == 2 else
                        [0]*len(df["component1"]),
                        label=velocity,
                        )

        plt.legend()
        plt.xlabel("Component 1")
        
        if nb_components == 2:
            plt.ylabel("Component 2")
        
    return technique, scaler


class SiameseNetwork(nn.Module):
    """
    Siamese Network class
    """
    def __init__(self, input_size: int):
        """Constructor of the class

        Args:
            input_size (int): Size of the input
            params (str): Path to the parameters file
        """        
        super(SiameseNetwork, self).__init__()
        
        # Define the architecture of the network
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network

        Args:
            x (torch.Tensor): Input of the network

        Returns:
            torch.Tensor: Output of the network
        """        
        # Apply the network to the input
        x = self.mlp(x)
        
        return x


def apply_model(features, model, params, device):
    
    # Load the weights of the model
    model.load_state_dict(torch.load(params))

    # Configure the model for testing
    model.eval()
    
    with torch.no_grad():
        
        features = torch.from_numpy(features).float()
        
        costs = model(features)
    
    costs = costs.numpy()
    
    return costs
    


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Test the dimensionality reduction
    technique, scaler = dimensionality_reduction(
        "src/traversal_cost/datasets/dataset_200Hz_wrap_fft/",
        method="pca",
        nb_components=2)
    
    # # Create a dummy feature vector
    # features = np.array([0.01, 0.02, 0.03])
    
    # # Scale the features
    # features_scaled = scaler.transform(features[None, :])
    
    # # Apply the dimensionality reduction
    # features_reduced = technique.transform(features_scaled)
    
    # print(features_reduced)
    
    plt.show()
