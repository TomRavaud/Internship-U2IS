import numpy as np
from tabulate import tabulate
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import io
from PIL import Image

# Import custom packages
import traversalcost.features
import params.traversal_cost
import traversalcost.fourier


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


def compute_traversal_costs(dataset: str,
                            cost_function: callable,
                            to_tensor: bool=False,
                            standardize: callable=None,
                            labels_file: str="labels.csv") -> pd.DataFrame:
    """Compute the traversal cost of each sample in a dataset

    Args:
        dataset (str): Path to the dataset
        cost_function (callable): Function used to compute the cost of a
        sample
        to_tensor (bool, optional): If True, convert the cost to a tensor.
        Defaults to False.
        standardize (callable, optional): Function used to standardize the
        features. Defaults to None.

    Returns:
        dataframe: A dataframe containing the terrain classes, the linear
        velocities of the robot and the traversal costs
    """    
    # Read the csv file containing the labels
    labels_df = pd.read_csv(dataset + labels_file,
                            converters={"id": str})
    
    # Add an empty column to the dataframe
    labels_df["cost"] = ""
    
    for i in range(len(labels_df.index)):
        # Get the id of the current sample
        id = labels_df["id"][i]
        
        # Load the features of the current sample
        features = np.load(dataset + "features/" + str(id) + ".npy")
        
        # Standardize the features if required
        if standardize:
            features = standardize(features)

        # Convert the features to a tensor if required and add a dimension
        if to_tensor:
            features = torch.from_numpy(features).float()
            features = features.unsqueeze_(0)
        
        # Compute the cost of the current sample
        cost = cost_function(features)
        
        # Convert the cost to a float if required
        if to_tensor:
            cost = cost.item()
        
        # Store the cost in the dataframe
        labels_df.at[i, "cost"] = cost
        
    # Extract the terrain classes, the linear velocities and the costs
    costs_df = labels_df[["terrain_class",
                          "linear_velocity",
                          "cost"]]
    
    return costs_df


def display_traversal_costs(costs_df: pd.DataFrame) -> Image:
    """Display the traversal costs of samples. Each terrain class is
    represented by a different color. The linear velocity is represented on
    the x-axis and the traversal cost on the y-axis.

    Args:
        costs_df (pd.Dataframe): A dataframe containing the terrain classes, the
        linear velocities of the robot and the traversal costs
        (headers: "terrain_class", "linear_velocity", "cost")
    
    Returns:
        Image: An image of the figure
    """
    # Get the list of the terrain classes
    labels_unique = list(set(costs_df["terrain_class"]))
    
    # Open a figure
    figure = plt.figure()

    # Go through the labels
    for label in labels_unique:
        
        # Get the samples of the current terrain class
        df = costs_df[costs_df["terrain_class"] == label]
        
        # If a color is specified for the current terrain class, use it
        if params.traversal_cost.colors.get(label):
            plt.scatter(df["linear_velocity"],
                        df["cost"],
                        label=label.replace("_", "\_"),
                        color=params.traversal_cost.colors[label])
        # Otherwise, use the default color
        else:
            plt.scatter(df["linear_velocity"],
                        df["cost"],
                        label=label.replace("_", "\_"))

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Traversal cost")
    
    # Converts the figure to an image
    image = io.BytesIO()
    figure.savefig(image, format="png", bbox_inches="tight")
    image.seek(0)
    
    # Create a PIL image from the image stream
    image = Image.open(image)
    
    return image


def display_traversal_costs_whiskers(costs_df: pd.DataFrame) -> Image:
    """Display the traversal costs of samples using whiskers.
    Each terrain class is represented by a different color.
    The linear velocity is represented on the x-axis and the
    traversal cost on the y-axis.

    Args:
        costs_df (pd.DataFrame): A dataframe containing the terrain classes,
        the linear velocities of the robot and the traversal costs

    Returns:
        Image: An image of the figure
    """
    
    # Get the list of the terrain classes
    labels_unique = list(set(costs_df["terrain_class"]))
    
    # Get the list of the linear velocities
    velocities_unique = list(set(costs_df["linear_velocity"]))
    
    # Open a figure
    fig, ax = plt.subplots()
    
    # Create a list of handles for the legend
    handles = []
    
    # Go through the labels
    for label in labels_unique:
        
        # Get the samples of the current terrain class
        df = costs_df[costs_df["terrain_class"] == label]
        
        # Add a handle for the current terrain class
        handles.append(mpatches.Patch(color=params.traversal_cost.colors[label],
                                      label=label.replace("_", "\_")))
        
        # Go through the linear velocities
        for velocity in velocities_unique:
            
            # Set the properties of the boxplot
            boxprops = dict(
                # facecolor=params.traversal_cost.colors[label],
                facecolor="white",
                color=params.traversal_cost.colors[label],
                )
            medianprops = dict(
                color=params.traversal_cost.colors[label],
                )
            capprops = dict(
                color=params.traversal_cost.colors[label],
                )
            whiskerprops = dict(
                color=params.traversal_cost.colors[label],
                )
            
            # print(type(df[df["linear_velocity"] == velocity]["cost"].values))
            
            # Plot the boxplot
            ax.boxplot(
                list(df[df["linear_velocity"] == velocity]["cost"]),
                notch=0,
                sym="",
                positions=[velocity],
                patch_artist=True,
                boxprops=boxprops,
                medianprops=medianprops,
                capprops=capprops,
                whiskerprops=whiskerprops,
                )
        
    # Set the limits of the axes
    ax.set_xlim(np.min(velocities_unique) - 0.2,
                np.max(velocities_unique) + 0.2)
    
    # Set the ticks of the axes
    ax.set_xticklabels(velocities_unique)
    
    # Set the legend
    ax.legend(handles=handles,
              bbox_to_anchor=(1, 1),
              loc="upper left")
    
    # Set the labels of the axes
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Traversal cost")
    
    # Converts the figure to an image
    image = io.BytesIO()
    fig.savefig(image, format="png", bbox_inches="tight")
    image.seek(0)
    
    # Create a PIL image from the image stream
    image = Image.open(image)
    
    return image


def display_confidence_intervals(costs_df: pd.DataFrame) -> Image:
    """Display the traversal costs of samples and the confidence interval.
    Each terrain class is represented by a different color.
    The linear velocity is represented on the x-axis and the traversal cost
    on the y-axis.

    Args:
        costs_df (pd.Dataframe): A dataframe containing the terrain classes, the
        linear velocities of the robot and the traversal costs
        (headers: "terrain_class", "linear_velocity", "cost")
    
    Returns:
        Image: An image of the figure
    """
    # Get the list of the terrain classes
    labels_unique = list(set(costs_df["terrain_class"]))
    
    # Get the list of the linear velocities (the list is sorted for
    # visualization purposes)
    velocities_unique = sorted(list(set(costs_df["linear_velocity"])))
    
    # Open a figure
    figure = plt.figure()
    
    # Set some parameters for the plot
    MARKER_SIZE = 8
    OPACITY = 0.2
    MARKER = "--."

    # Go through the labels
    for label in labels_unique:
        
        # Get the samples of the current terrain class
        df = costs_df[costs_df["terrain_class"] == label]
        
        # Create arrays to store the mean costs and the confidence intervals
        mean_costs = np.zeros(len(velocities_unique))
        confidence = np.zeros(len(velocities_unique))
        
        for i, velocity in enumerate(velocities_unique):
            
            df_velocity = df[df["linear_velocity"] == velocity]
            
            # Compute the mean cost
            mean_costs[i] = np.mean(df_velocity["cost"])
            
            # Compute the 99.7% confidence interval
            confidence[i] = 3*np.std(df_velocity["cost"])\
                            /np.sqrt(len(df_velocity["cost"]))
        
        # If a color is specified for the current terrain class, use it
        if params.traversal_cost.colors.get(label):
            plt.plot(velocities_unique,
                     mean_costs,
                     MARKER,
                     markersize=MARKER_SIZE,
                     label=label.replace("_", "\_"),
                     color=params.traversal_cost.colors[label])
            plt.fill_between(velocities_unique,
                             mean_costs - confidence,
                             mean_costs + confidence,
                             color=params.traversal_cost.colors[label],
                             alpha=OPACITY)
            
        # Otherwise, use the default color
        else:
            plt.plot(velocities_unique,
                     mean_costs,
                     MARKER,
                     markersize=MARKER_SIZE,
                     label=label.replace("_", "\_"))
            plt.fill_between(velocities_unique,
                             mean_costs - confidence,
                             mean_costs + confidence,
                             alpha=OPACITY)

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Traversal cost")
    
    # Converts the figure to an image
    image = io.BytesIO()
    figure.savefig(image, format="png", bbox_inches="tight")
    image.seek(0)
    
    # Create a PIL image from the image stream
    image = Image.open(image)
    
    return image


def modulo_wrap(signal: list, N: int) -> list:
    """Wrap a signal by splitting it into blocks of given length and
    summing the blocks. If the length of the signal is not a multiple of
    the block length, the last block is padded with zeros.
    (See https://www.ece.rutgers.edu/~orfanidi/intro2sp/ for more details)

    Args:
        signal (list): Original signal
        N (int): Length of the blocks

    Returns:
        list: Wrapped signal
    """    
    # Get the length of the signal
    L = len(signal)
    
    # Create a list of zeros of length N to store the wrapped signal
    wrapped_signal = [0] * N
    
    # Compute the quotient and the remainder of the euclidean division
    # of L by N 
    M = L // N
    r = L % N
    
    # Wrap the original signal
    for n in range(N):
        
        # Non-zero part of last block
        # (if L < N, this is the only block)
        if n < r:
            wrapped_signal[n] = signal[M*N + n]
        
        # Pad N - L zeros at end if L < N
        else:
            wrapped_signal[n] = 0 

        # Remaining blocks
        # (if L < N, this loop is skipped)
        for m in range(M - 1, -1, -1):
            wrapped_signal[n] += signal[m*N + n]
    
    return wrapped_signal


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Test the functions
    FEATURES = {"function": traversalcost.features.variance,
                "params_roll_rate": {},
                "params_pitch_rate": {},
                "params_vertical_acceleration": {}}
    
    # print(generate_description(FEATURES))
    
    costs_df = compute_traversal_costs(
        dataset="src/traversal_cost/datasets/dataset_40Hz_variance/",
        cost_function=np.mean
        )
    
    # costs = display_traversal_costs(costs_df)
    
    # whiskers = display_traversal_costs_whiskers(costs_df)
    # Convert the image to grayscale
    # image_pgm = image.convert("L")
    # image_pgm.save("traversal_cost_whiskers.pgm")
    
    display_confidence_intervals(costs_df)
    

    # Save the image
    # image.save("plot.png", "PNG")
     
    # Modulo-N reduction test
    N = 50
    
    # Create dummy signals and wrap them
    signal = np.random.rand(100)
    signal_wrapped = modulo_wrap(signal, N)
    
    signal2 = np.random.rand(137)
    signal2_wrapped = modulo_wrap(signal2, N)
    
    # Apply the FFT to the signals and plot the results
    # traversalcost.fourier.fft(signal, 40, plot=True)
    # traversalcost.fourier.fft(signal_wrapped, 40, plot=True)
    # traversalcost.fourier.fft(signal2, 40, plot=True)
    
    plt.show()
