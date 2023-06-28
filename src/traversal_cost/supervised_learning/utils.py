import numpy as np
import pandas


def compute_mean_std(dataset: str) -> tuple:
    """Compute the mean and the standard deviation of the features of the
    dataset

    Args:
        dataset (str): Path to the dataset

    Returns:
        tuple: mean and standard deviation
    """
    
    # Read the csv file
    df = pandas.read_csv(dataset + "traversal_costs_train.csv",
                         converters={'id' : str})
    
    # Get the number of samples
    nb_samples = len(df)
    
    # Get the number of features per sample
    nb_features = len(
        np.load(dataset + "features/" + df.loc[0, "id"] + ".npy"))
    
    # Create a numpy array to store the features of all the samples used in
    # the training
    features = np.zeros((nb_samples, nb_features))
    
    # Fill the array of features
    for i in range(nb_samples):
        
        # Get the id of the signal
        id = df.loc[i, "id"]
        
        # Get the features of the signal
        features[i, :] = np.load(dataset + "features/" + id + ".npy")
    
    # Compute the mean and the standard deviation of the features
    mean, std = np.mean(features, axis=0), np.std(features, axis=0)
    
    return mean, std


class Standardize:
    """
    Class to standardize and de-standardize a value
    """
    
    def __init__(self, mean: float, std: float) -> None:
        """Constructor of the class

        Args:
            mean (float): Mean
            std (float): Standard deviation
        """
        self.mean = mean
        self.std = std
        
    def standardize(self, x: float) -> float:
        """Standardize a value

        Args:
            x (float): Value to standardize

        Returns:
            float: The standardized value
        """
        return (x - self.mean) / self.std
    
    def destandardize(self, x: float) -> float:
        """De-standardize a value

        Args:
            x (float): Value to de-standardize

        Returns:
            float: The de-standardized value
        """        
        return x * self.std + self.mean
        

# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    mean, std = compute_mean_std(
        dataset="src/traversal_cost/datasets/dataset_to_delete/")
    
    # print("Mean: ", mean)
    # print("Std: ", std)
    
    # Generate a dummy signal
    signal = np.random.rand(len(mean))
    
    # Instantiate the standardize class
    standardize = Standardize(mean, std)
    
    # Standardize the signal
    signal = standardize.standardize(signal)
    
    # De-standardize the signal
    signal = standardize.destandardize(signal)
