"""
A script to tune the hyperparameters of the Siamese Network using Optuna.
"""

# Import packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch import optim
import optuna
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "figure.figsize": (10,5),
})

# Import custom packages and modules
import params.supervised_learning
from dataset import SupervisedLearningDataset
from model import SupervisedNetwork
from train import train
from validate import validate
from test import test 
from result import parameters_table, generate_log
from utils import compute_mean_std, Standardize


def objective(trial):
    """Objective function for Optuna.

    Args:
        trial (_type_): A trial is a single call of the objective function.

    Returns:
        float: The value of the objective function.
    """    
    
    print("\nTrial", trial.number) 
    
    # Select parameter uniformaly within the range
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    # wd = trial.suggest_float("wd", 1e-8, 1e-1, log=True)
    batch_size = trial.suggest_int("batch\_size", 1, 64)
    
    # Load learning parameters
    LEARNING_PARAMS = params.supervised_learning.LEARNING
    
    # Update the learning parameters
    LEARNING_PARAMS.update({
        # "weight_decay": wd,
        "learning_rate": lr,
        "batch_size": batch_size
        })
    
    # Set the path to the dataset
    DATASET = "src/traversal_cost/datasets/dataset_to_delete/"
    
    # Standardize the data
    mean, std = compute_mean_std(DATASET)
    # Instantiate the standardize class
    standardize = Standardize(mean, std)
    
    # Create a Dataset for training
    train_set = SupervisedLearningDataset(
        DATASET + "traversal_costs_train.csv",
        DATASET + "/features",
        transform=standardize.standardize)
    
    # Create a Dataset for validation
    # (same as training here since no transformation is applied to the data,
    # train and validation sets will be split later)
    val_set = SupervisedLearningDataset(
        DATASET + "traversal_costs_train.csv",
        DATASET + "/features",
        transform=standardize.standardize)
    
    # Create a Dataset for testing
    test_set = SupervisedLearningDataset(
        DATASET +"traversal_costs_test.csv",
        DATASET + "/features",
        transform=standardize.standardize)

    # Set the train dataset size
    train_size = params.supervised_learning.TRAIN_SIZE/(1-params.supervised_learning.TEST_SIZE)

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
    model = SupervisedNetwork(input_size=nb_input_features).to(device=device)
    
    # Create a loss function
    criterion = nn.MSELoss()
    
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_PARAMS["learning_rate"],
                          momentum=LEARNING_PARAMS["momentum"],
                          weight_decay=LEARNING_PARAMS["weight_decay"])

    # Create tensors to store the loss values
    loss_values = torch.zeros(2, LEARNING_PARAMS["nb_epochs"])
    
    # Train the model for multiple epochs
    best_val_loss = np.inf
    best_epoch = 0
    
    # Loop over the epochs
    for epoch in range(LEARNING_PARAMS["nb_epochs"]):

        # Training
        train_loss = train(model,
                           device,
                           train_loader,
                           optimizer,
                           criterion,
                           epoch)

        # Validation
        val_loss = validate(model,
                            device,
                            val_loader,
                            criterion,
                            epoch) 

        print("Train loss: ", train_loss)
        print("Validation loss: ", val_loss)

        # Store the computed losses
        loss_values[0, epoch] = train_loss
        loss_values[1, epoch] = val_loss
        
        # Abort the training if a NaN value is encountered (exploding gradient)
        if np.isnan(val_loss):
            print("NaN value encountered, stopping the training")
            return
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    
    # Test the model
    test_loss = test(model,
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
                        "/src/traversal_cost/supervised_learning/logs_optuna3/_" +\
                        str(trial.number)
 
    # Generate the log directory
    generate_log(dataset_directory=DATASET,
                 results_directory=results_directory,
                 test_loss=test_loss,
                 parameters_table=params_table,
                 model=model,
                 loss_values=loss_values,
                 standardize=standardize.standardize,
                 labels_file="traversal_costs_test.csv")
    
    # Close all the previously opened figures
    plt.close("all")

    return best_val_loss


# Set the number of trials
NB_TRIALS = 200

# Ignore warnings
# warnings.filterwarnings("ignore")

# Samplers: GridSampler, RandomSampler, TPESampler, CmaEsSampler,
# PartialFixedSampler, NSGAIISampler, QMCSampler
# Pruners: MedianPruner, NopPruner, PatientPruner, PercentilePruner,
# SuccessiveHalvingPruner, HyperbandPruner, ThresholdPruner
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    storage = "sqlite:///src/traversal_cost/supervised_learning/logs_optuna3/supervised_cost.db",
    study_name = "supervised_learning"
    )

study.optimize(objective, n_trials=NB_TRIALS)
