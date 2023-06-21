from tabulate import tabulate
import os
import cv2
import torch
from typing import List, Any
import PIL
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Import custom packages and modules
import params.learning


def parameters_table(dataset: str,
                     learning_params: dict) -> List[List[Any]]:
    """Generate a table containing the parameters used to train the network
    
    Args:
        dataset (str): The path to the dataset
        learning_params (dict): The parameters used to train the network

    Returns:
        table: The table of parameters
    """
    # Generate the description of the training parameters
    data = [
        [
            "Dataset",
            "Train size",
            "Validation size",
            "Test size",
            "Batch size",
            "Nb epochs",
            "Learning rate",
            "Weight decay",
            "Momentum",
        ],
        [
            dataset.split("/")[-2],
            params.learning.TRAIN_SIZE,
            params.learning.VAL_SIZE,
            params.learning.TEST_SIZE,
            learning_params["batch_size"],
            learning_params["nb_epochs"],
            learning_params["learning_rate"],
            learning_params["weight_decay"],
            learning_params["momentum"],
        ],
    ]
    
    # Generate the table
    table = tabulate(data,
                     headers="firstrow",
                     tablefmt="fancy_grid",
                     maxcolwidths=20,
                     numalign="center",)
    
    return table


def generate_log(results_directory: str,
                 test_regression_loss: float,
                 test_accuracy: float,
                 parameters_table: List[List[Any]],
                 model: torch.nn.Module,
                 regression_loss_values: torch.Tensor,
                 accuracy_values: torch.Tensor,
                 test_losses_loss: list,
                 test_losses_uncertainty: list) -> None:
    """Create a directory to store the results of the training and save the
    results in it

    Args:
        results_directory (str): Path to the directory where the results will
        be stored
        test_regression_loss (float): Test loss
        test_accuracy (float): Test accuracy
        parameters_table (table): Table of parameters
        model (nn.Module): The network
        regression_loss_values (Tensor): Regression loss values
        accuracy_values (Tensor): Accuracy values
        test_losses_loss (list): Test loss values when removing samples with
        the highest regression loss
        test_losses_uncertainty (list): Test loss values when removing samples
        with the highest uncertainty
    """    
    # Create the directory
    os.mkdir(results_directory)
    
    # Open a text file
    test_loss_file = open(results_directory + "/test_results.txt", "w")
    # Write the test loss in it
    test_loss_file.write(f"Test regression loss: {test_regression_loss}\n")
    # Write the test accuracy in it
    test_loss_file.write(f"Test accuracy: {test_accuracy}")
    # Close the file
    test_loss_file.close()
    
    # Open a text file
    parameters_file = open(results_directory + "/parameters_table.txt", "w")
    # Write the table of learning parameters in it
    parameters_file.write(parameters_table)
    # Close the file
    parameters_file.close()
    
    # Open a text file
    network_file = open(results_directory + "/network.txt", "w")
    # Write the network in it
    print(model, file=network_file)
    # Close the file
    network_file.close()
    
    # Create and save the learning curve
    train_losses = regression_loss_values[0]
    val_losses = regression_loss_values[1]

    plt.figure()

    plt.plot(train_losses, "b", label="train loss")
    plt.plot(val_losses, "r", label="validation loss")

    plt.legend()
    plt.xlabel("Epoch")
    
    plt.savefig(results_directory + "/learning_curve.png")
    
    # Create and save the accuracy curve
    train_accuracies = accuracy_values[0]
    val_accuracies = accuracy_values[1]
    
    plt.figure()

    plt.plot(train_accuracies, "b", label="train accuracy")
    plt.plot(val_accuracies, "r", label="validation accuracy")

    plt.legend()
    plt.xlabel("Epoch")
    
    plt.savefig(results_directory + "/accuracy_curve.png")
    
    plt.figure()
    
    plt.plot(range(0, 100, 10),
             test_losses_loss,
             "bo--",
             label="removing samples with highest regression loss",
             markersize=4)
    plt.plot(range(0, 100, 10),
             test_losses_uncertainty,
             "ro--",
             label="removing samples with highest uncertainty",
             markersize=4)

    plt.legend(loc="upper right")
    plt.xlabel("Percentage of samples removed")
    plt.ylabel("Regression error (MSE)")
    
    plt.savefig(results_directory + "/uncertainty_relevance.png")
    
    # Save the model parameters
    torch.save(model.state_dict(),
               results_directory + "/" + params.learning.PARAMS_FILE)


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Test the functions
    print(parameters_table())
