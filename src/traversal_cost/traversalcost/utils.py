import numpy as np
from tabulate import tabulate
import inspect
import pandas as pd
import matplotlib.pyplot as plt

# Import custom packages
import traversalcost.features
import params.traversal_cost


def get_features(roll_rate_values,
                 pitch_rate_values,
                 vertical_acceleration_values,
                 dict):
    """Extract features from IMU signals and concatenate them

    Args:
        roll_rate_values (list): List of roll rate values
        pitch_rate_values (list): List of pitch rate values
        vertical_acceleration_values (list): List of vertical acceleration
        values
        dict (dictionary): Dictionary containing the function to be applied to
        each signal and the parameters to be passed to the function

    Returns:
        ndarray (n,): Concatenated features
    """
    # Compute the features from the IMU signals
    features = [
        dict["function"](roll_rate_values,
                         **dict["params_roll_rate"]),
        dict["function"](pitch_rate_values,
                         **dict["params_pitch_rate"]),
        dict["function"](vertical_acceleration_values,
                         **dict["params_vertical_acceleration"])
    ]
    
    # Concatenate the features
    features = np.concatenate(features)
    
    return features

def generate_description(dict):
    """Generate a description of a function used to extract features from IMU
    signals

    Args:
        dict (dictionary): Dictionary containing the function to be applied to
        each signal and the parameters to be passed to the function

    Returns:
        table: A table containing the description of the function
    """
    # Generate a dummy signal and extract the features
    dummy_signal = np.random.rand(100)
    dummy_features = dict["function"](dummy_signal)
    
    # Get the default arguments of the function
    signature = inspect.signature(dict["function"])
    default_args = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    
    # Copy and update the default arguments with the roll rate parameters
    args_roll_rate = default_args.copy()
    args_roll_rate.update(dict["params_roll_rate"])
    
    # Copy and update the default arguments with the pitch rate parameters
    args_pitch_rate = default_args.copy()
    args_pitch_rate.update(dict["params_pitch_rate"])
    
    # Copy and update the default arguments with the vertical acceleration
    # parameters
    args_vertical_acceleration = default_args.copy()
    args_vertical_acceleration.update(dict["params_vertical_acceleration"])
    
    # Generate the description of the function
    data = [
        [
            "Function",
            "Description",
            "Nb features/signal",
            "Params roll rate",
            "Params pitch rate",
            "Params vertical acceleration",
        ],
        [
            dict["function"].__name__,
            (dict["function"].__doc__).split("\n\n")[0],
            len(dummy_features),
            args_roll_rate,
            args_pitch_rate,
            args_vertical_acceleration,
        ],
    ]
    
    # Generate the table
    table = tabulate(data,
                     headers="firstrow",
                     tablefmt="fancy_grid",
                     maxcolwidths=20,
                     numalign="center",)
    
    return table


def compute_traversal_costs(dataset,
                            cost_function):
    """Compute the traversal cost of each sample in a dataset

    Args:
        dataset (string): Path to the dataset
        cost_function (function): Function used to compute the cost of a
        sample

    Returns:
        dataframe: A dataframe containing the terrain classes, the linear
        velocities of the robot and the traversal costs
    """    
    # Read the csv file containing the labels
    labels_df = pd.read_csv(dataset + "labels.csv",
                            converters={"id": str})
    
    # Add an empty column to the dataframe
    labels_df["cost"] = ""
    
    for i in range(len(labels_df.index)):
        # Get the id of the current sample
        id = labels_df["id"][i]
        
        # Load the features of the current sample
        features = np.load(dataset + "features/" + str(id) + ".npy")
        
        # Compute the cost of the current sample
        cost = cost_function(features)
        
        # Store the cost in the dataframe
        labels_df.at[i, "cost"] = cost
        
    # Extract the terrain classes, the linear velocities and the costs
    costs_df = labels_df[["terrain_class",
                          "linear_velocity",
                          "cost"]]
    
    return costs_df

def display_traversal_costs(costs_df):
    """Display the traversal costs of samples. Each terrain class is
    represented by a different color. The linear velocity is represented on
    the x-axis and the traversal cost on the y-axis.

    Args:
        costs_df (dataframe): A dataframe containing the terrain classes, the
        linear velocities of the robot and the traversal costs
        (headers: "terrain_class", "linear_velocity", "cost")
    """
    # Get the list of the terrain classes
    labels_unique = list(set(costs_df["terrain_class"]))
    
    # Open a figure
    plt.figure()
    
    # Go through the labels
    for label in labels_unique:
        
        # Get the samples of the current terrain class
        df = costs_df[costs_df["terrain_class"] == label]
        
        # If a color is specified for the current terrain class, use it
        if params.traversal_cost.colors.get(label):
            plt.scatter(df["linear_velocity"],
                        df["cost"],
                        label=label,
                        color=params.traversal_cost.colors[label])
        # Otherwise, use the default color
        else:
            plt.scatter(df["linear_velocity"],
                        df["cost"],
                        label=label)

    plt.legend()

    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Traversal cost")


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Test the functions
    FEATURES = {"function": traversalcost.features.variance,
                "params_roll_rate": {},
                "params_pitch_rate": {},
                "params_vertical_acceleration": {}}
    
    print(generate_description(FEATURES))
    
    costs_df = compute_traversal_costs(
        dataset="src/traversal_cost/datasets/dataset_write/",
        cost_function=np.mean
        )
    
    display_traversal_costs(costs_df)
    plt.show()
