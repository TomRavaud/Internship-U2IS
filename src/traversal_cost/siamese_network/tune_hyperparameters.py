"""
A script to tune the hyperparameters of the Siamese Network using Optuna.
"""

# Import packages
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch import optim
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to be able to save
                       # figures when running the script on a remote server
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "figure.figsize": (10,5),
})
import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
import warnings
import os
from datetime import datetime

# Import custom packages and modules
import params.siamese
import traversalcost.utils
from dataset import SiameseNetworkDataset
from model import SiameseNetwork
from loss import SiameseLoss
from train import train
from validate import validate
from test import test
from result import parameters_table, generate_log


def objective(trial):
    """Objective function for Optuna.

    Args:
        trial (_type_): A trial is a single call of the objective function.

    Returns:
        float: The value of the objective function.
    """    
    
    print("\nTrial", trial.number) 
    
    # Select parameter uniformaly within the range
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # margin = trial.suggest_float("margin", 1e-2, 1e1, log=True)
    # nb_layers = trial.suggest_int("nb\_layers", 1, 5)
    # nb_layers = 4
    # output_activation = trial.suggest_categorical("output\_activation",
    #                                               ["sigmoid", "relu"])
    wd = trial.suggest_float("wd", 1e-10, 1e-1, log=True)
    
    # Load learning parameters
    LEARNING_PARAMS = params.siamese.LEARNING
    
    # Update the learning parameters
    LEARNING_PARAMS.update({"weight_decay": wd})
    
    # Set the path to the dataset
    DATASET = "src/traversal_cost/datasets/dataset_40Hz_dwt_hard/"

    # Create a Dataset for training
    train_set = SiameseNetworkDataset(
        pairs_file=DATASET+"pairs_train.csv",
        features_directory=DATASET+"features",
    )

    # Create a Dataset for validation
    # (same as training here since no transformation is applied to the data,
    # train and validation sets will be split later)
    val_set = SiameseNetworkDataset(
        pairs_file=DATASET+"pairs_train.csv",
        features_directory=DATASET+"features",
    )

    # Create a Dataset for testing
    test_set = SiameseNetworkDataset(
        pairs_file=DATASET+"pairs_test.csv",
        features_directory=DATASET+"features",
    )

    # Set the train dataset size
    train_size = params.siamese.TRAIN_SIZE/(1-params.siamese.TEST_SIZE)

    # Splits train data indices into train and validation data indices
    train_indices, val_indices = train_test_split(range(len(train_set)),
                                                  train_size=train_size)

    # Extract the corresponding subsets of the train dataset
    train_set = Subset(train_set, train_indices)
    val_set = Subset(val_set, val_indices)


    # Combine a dataset and a sampler, and provide an iterable over the dataset
    # (setting shuffle argument to True calls a RandomSampler, and avoids to
    # have to create a Sampler object)
    train_loader = DataLoader(
        train_set,
        batch_size=LEARNING_PARAMS["batch_size"],
        shuffle=True,
        num_workers=12,  # Asynchronous data loading and augmentation
        pin_memory=True,  # Increase the transferring speed of the data
                          # to the GPU
    )

    val_loader = DataLoader(
        val_set,
        batch_size=LEARNING_PARAMS["batch_size"],
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=LEARNING_PARAMS["batch_size"],
        shuffle=False,  # SequentialSampler
        num_workers=12,
        pin_memory=True,
    )
    
    # Use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    # Create a model
    nb_input_features = len(train_set[0][0])
    
    # layers = []
    
    # for i in range(nb_layers):
        
    #     nb_out_features = trial.suggest_int(f"nb\_units\_l{i}", 4, 128)
        
    #     layers.append(torch.nn.Linear(nb_input_features, nb_out_features))
    #     layers.append(torch.nn.ReLU())
        
    #     nb_input_features = nb_out_features
        
    # layers.append(torch.nn.Linear(nb_input_features, 1))
    
    # if output_activation == "sigmoid":
    #     layers.append(torch.nn.Sigmoid())
    # elif output_activation == "relu":
    #     layers.append(torch.nn.ReLU())
    
    # model = torch.nn.Sequential(*layers).to(device=device)
    
    model = SiameseNetwork(input_size=nb_input_features).to(device=device)
    
    
    # Create a loss function
    criterion = SiameseLoss(margin=
                            LEARNING_PARAMS["margin"]).to(device=device)
    
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_PARAMS["learning_rate"],
                          momentum=LEARNING_PARAMS["momentum"],
                          weight_decay=LEARNING_PARAMS["weight_decay"])

    # Create tensors to store the loss values
    loss_values = torch.zeros(2, LEARNING_PARAMS["nb_epochs"])
    
    # Create tensors to store the accuracy values
    accuracy_values = torch.zeros(2, LEARNING_PARAMS["nb_epochs"])

    # Define the number of epochs to wait before stopping the training
    # if the validation loss does not improve
    PATIENCE = 10
    
    # Train the model for multiple epochs
    best_val_accuracy = 0.
    best_epoch = 0
    
    # Loop over the epochs
    for epoch in range(LEARNING_PARAMS["nb_epochs"]):

        # Training
        train_loss, train_accuracy = train(model,
                                           device,
                                           train_loader,
                                           optimizer,
                                           criterion,
                                           epoch)

        # Validation
        val_loss, val_accuracy = validate(model,
                                          device,
                                          val_loader,
                                          criterion,
                                          epoch) 

        print("Train loss: ", train_loss)
        print("Validation loss: ", val_loss)
        print("Train accuracy: ", train_accuracy)
        print("Validation accuracy: ", val_accuracy)

        # Store the computed losses
        loss_values[0, epoch] = train_loss
        loss_values[1, epoch] = val_loss
        
        # Store the computed accuracies
        accuracy_values[0, epoch] = train_accuracy
        accuracy_values[1, epoch] = val_accuracy
        
        # Early stopping based on validation accuracy: stop the training if
        # the accuracy has not improved for the last 5 epochs
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
        
        elif epoch - best_epoch >= PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Test the model
    test_loss, test_accuracy = test(model,
                                    device,
                                    test_loader,
                                    criterion)
    
    # Get the learning parameters table
    params_table = parameters_table(dataset=DATASET,
                                    learning_params=LEARNING_PARAMS)

    # Get the absolute path of the current directory
    directory = os.path.abspath(os.getcwd())
    
    # Set the path to the results directory
    results_directory = directory +\
                        "/src/traversal_cost/siamese_network/logs_optuna_wd/_" +\
                        str(trial.number)
 
    # Generate the log directory
    generate_log(dataset_directory=DATASET,
                 results_directory=results_directory,
                 test_loss=test_loss,
                 test_accuracy=test_accuracy,
                 parameters_table=params_table,
                 model=model,
                 loss_values=loss_values,
                 accuracy_values=accuracy_values)

    return best_val_accuracy


# Set the number of trials
NB_TRIALS = 30

# Ignore warnings
warnings.filterwarnings("ignore")

# Samplers: GridSampler, RandomSampler, TPESampler, CmaEsSampler,
# PartialFixedSampler, NSGAIISampler, QMCSampler
# Pruners: MedianPruner, NopPruner, PatientPruner, PercentilePruner,
# SuccessiveHalvingPruner, HyperbandPruner, ThresholdPruner
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    # pruner=optuna.pruners.MedianPruner()
    )
study.optimize(objective, n_trials=NB_TRIALS)

# Get the best parameters
best_params = study.best_params
# found_learning_rate = best_params["lr"]
# found_margin = best_params["margin"]

# print("Found learning rate: ", found_learning_rate)
# print("Found margin: ", found_margin)

# Close all the previously opened figures
plt.close("all")

# Plot functions
axes_optimization_history = plot_optimization_history(study)
axes_optimization_history.loglog()
axes_optimization_history.legend()

# Save the figure
axes_optimization_history.get_figure().savefig("optimization_history.png")

# # When pruning
# axes_intermediate_values = plot_intermediate_values(study)

axes_parallel_coordinate = plot_parallel_coordinate(study)

# Save the figure
axes_parallel_coordinate.get_figure().savefig("parallel_coordinate.png")

# # Should be at least 2 parameters to see something
# plot_contour(study)

# Plot the importance of the parameters
axes_param_importances = plot_param_importances(study)

# Save the figure
axes_param_importances.get_figure().savefig("param_importances.png")

# plt.show()
