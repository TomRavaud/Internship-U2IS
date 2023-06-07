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
import params.siamese
import traversalcost.utils


def parameters_table(dataset: str,
                     learning_params: dict) -> List[List[Any]]:
    """Generate a table containing the parameters used to train the Siamese
    network

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
            "Margin",
            "Learning rate",
            "Weight decay",
            "Momentum",
        ],
        [
            dataset.split("/")[-2],
            params.siamese.TRAIN_SIZE,
            params.siamese.VAL_SIZE,
            params.siamese.TEST_SIZE,
            learning_params["batch_size"],
            learning_params["nb_epochs"],
            learning_params["margin"],
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

def generate_log(dataset_directory: str,
                 results_directory: str,
                 test_loss: float,
                 test_accuracy: float,
                 parameters_table: List[List[Any]],
                 model: torch.nn.Module,
                 loss_values: torch.Tensor,
                 accuracy_values: torch.Tensor):
    """Create a directory to store the results of the training and save the
    results in it

    Args:
        test_loss (float): Test loss
        parameters_table (table): Table of parameters
        model (nn.Module): Siamese network
        learning_curve (PIL Image): Learning curve image
        traversal_cost_graph (PIL Image): Traversal cost graph image
    """    
    # Create the directory
    os.mkdir(results_directory)
    
    # Open a text file
    test_loss_file = open(results_directory + "/test_results.txt", "w")
    # Write the test loss in it
    test_loss_file.write(f"Test loss: {test_loss}\n")
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
    train_losses = loss_values[0]
    val_losses = loss_values[1]

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
    
    # Compute the traversal costs from the features of the dataset
    costs_df = traversalcost.utils.compute_traversal_costs(
        dataset=dataset_directory,
        cost_function=model.to(device="cpu"),
        to_tensor=True)
    
    # Display the traversal costs
    cost_graph = traversalcost.utils.display_traversal_costs(costs_df)
    
    # Save the traversal cost graph
    cost_graph.save(results_directory + "/traversal_cost_graph.png", "PNG")
    
    # Display the whiskers
    cost_whiskers = traversalcost.utils.display_traversal_costs_whiskers(costs_df)
    
    # Save the whiskers
    cost_whiskers.save(results_directory + "/traversal_cost_whiskers.png", "PNG")
    
    # Save the model parameters
    torch.save(model.state_dict(),
               results_directory + "/" + params.siamese.PARAMS_FILE)


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Test the functions
    print(parameters_table())
