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
import traversal_cost_supervised
import traversalcost.fourier


def compute_traversal_costs(dataset,
                            cost_function,
                            only_test,
                            to_tensor=False
                            ):
    """Compute the traversal cost of each sample in a dataset

    Args:
        dataset (string): Path to the dataset
        cost_function (function): Function used to compute the cost of a
        sample
        to_tensor (bool, optional): If True, convert the cost to a tensor.
        Defaults to False.

    Returns:
        dataframe: A dataframe containing the terrain classes, the linear
        velocities of the robot and the traversal costs
    """    
    # Read the csv file containing the labels
    
    if only_test == False :
        

        labels_df = pd.read_csv(dataset + "labels.csv",
                            converters={"id": str})
    
        # Add an empty column to the dataframe
        labels_df["cost"] = ""
    
        for i in range(len(labels_df.index)):
            print(i)
            
            # Get the id of the current sample
            id = labels_df["id"][i]
            print(id)
        
            # Load the features of the current sample
            features = np.load(dataset + "features/" + str(id) + ".npy")
            
            # Convert the features to a tensor if required
            if to_tensor:
                features = torch.from_numpy(features).float()
            
            # Compute the cost of the current sample
            cost = cost_function(features)
            
            # Convert the cost to a float if required
            if to_tensor:
                cost = cost.item()
            
            # Store the cost in the dataframe
            labels_df.at[i, "cost"] = cost
        
        labels_df.to_csv('test_cost_list')

    if only_test == True :
        
        labels_df = pd.read_csv(dataset + "labels.csv",
                            converters={"id": str})
        
        test_data_labels_df =  pd.read_csv(dataset + "traversalcosts_test.csv",
                            converters={"id": str})
        
        # Add an empty column to the dataframe
        labels_df["cost"] = np.nan

        # Add custom indexes to have match between labels_df and test_data_labels_df
        labels_df.set_index('id', inplace=True)
        test_data_labels_df.set_index('id')
        
                        
        for i in range(len(test_data_labels_df.index)):
            
            # Get the id of the current sample
            id = test_data_labels_df["id"][i]
        
            # Load the features of the current sample
            features = np.load(dataset + "features/" + str(id) + ".npy")
            
            # Convert the features to a tensor if required
            if to_tensor:
                features = torch.from_numpy(features).float()
            
            # Compute the cost of the current sample
            cost = cost_function(features)
            
            # Convert the cost to a float if required
            if to_tensor:
                cost = cost.item()

            # Store the cost in the dataframe
            labels_df.at[id, "cost"] = cost
    
    #Drops the empty rows of the dataframe
    labels_df = labels_df.dropna()
    
    # Extract the terrain classes, the linear velocities and the costs
    costs_df = labels_df[["terrain_class",
                          "linear_velocity",
                          "cost"]]
    
    return costs_df


def display_traversal_costs(costs_df: pd.DataFrame, theoric, dictionnary) -> Image:
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
    plt.figure(figsize=(20,20)) #ICI

    # Go through the labels
    for label in labels_unique:
        
        # Get the samples of the current terrain class
        df = costs_df[costs_df["terrain_class"] == label]
        
        # If a color is specified for the current terrain class, use it
        
        if traversal_cost_supervised.colors.get(label):
            plt.scatter(df["linear_velocity"],
                        df["cost"],
                        label=label.replace("_", "\_"),
                        color=traversal_cost_supervised.colors[label])
            
            if theoric == True : 
            
                t = np.linspace(0, 1.0, 500)
                plt.plot(t, dictionnary[label](t),color=traversal_cost_supervised.colors[label])
            
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


def display_traversal_costs_whiskers(costs_df: pd.DataFrame, theoric, dictionnary) -> Image:
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
    
    # print(costs_df.info(verbose=True))
    # print(costs_df["cost"].detach().numpy().describe())
    
    # costs_df.astype({"cost": "float64"}).dtypes
    
    # Get the list of the terrain classes
    labels_unique = list(set(costs_df["terrain_class"]))
    
    # Get the list of the linear velocities
    velocities_unique = list(set(costs_df["linear_velocity"]))
    
    # Open a figure
    fig, ax = plt.subplots(figsize = (20,20))
    
    # Create a list of handles for the legend
    handles = []
    
    # Go through the labels
    for label in labels_unique:
        
        # Get the samples of the current terrain class
        df = costs_df[costs_df["terrain_class"] == label]
        
        # Add a handle for the current terrain class
        handles.append(mpatches.Patch(color=traversal_cost_supervised.colors[label],
                                      label=label.replace("_", "\_")))
        
        # Go through the linear velocities
        for velocity in velocities_unique:
            
            # Set the properties of the boxplot
            boxprops = dict(
                # facecolor=traversal_cost_supervised.colors[label],
                facecolor="white",
                color=traversal_cost_supervised.colors[label],
                )
            medianprops = dict(
                color=traversal_cost_supervised.colors[label],
                )
            capprops = dict(
                color=traversal_cost_supervised.colors[label],
                )
            whiskerprops = dict(
                color=traversal_cost_supervised.colors[label],
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
            
            if traversal_cost_supervised.colors.get(label):
                ax.scatter(df["linear_velocity"],
                        df["cost"],
                        label=label.replace("_", "\_"),
                        color=traversal_cost_supervised.colors[label])
            
            if theoric == True : 
            
                t = np.linspace(0, 1.0, 500)
                ax.plot(t, dictionnary[label](t),color=traversal_cost_supervised.colors[label])
        
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
    whiskers = display_traversal_costs_whiskers(costs_df)
    # Convert the image to grayscale
    # image_pgm = image.convert("L")
    # image_pgm.save("traversal_cost_whiskers.pgm")

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
