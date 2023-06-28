"""
A script to tune the hyperparameters of the network using Optuna.
"""

# Import packages
import torch
import torch.nn as nn
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
import warnings
import os
import numpy as np

# Import custom packages and modules
import params.learning
from dataset import TraversabilityDataset
import transforms
from model import ResNet18Velocity
from model2 import TraversabilityNet
from train import train
from validate import validate
from test import test
from result import parameters_table, generate_log
import uncertainty.functions
import uncertainty.evaluation


def objective(trial):
    """Objective function for Optuna.

    Args:
        trial (_type_): A trial is a single call of the objective function.

    Returns:
        float: The value of the objective function.
    """    
    print("\nTrial", trial.number)
    
    # Select parameter uniformaly within the range
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # wd = trial.suggest_float("wd", 1e-8, 1e-2, log=True)
    batch_size = trial.suggest_int("batch\_size", 1, 128)
    nb_fc_features = trial.suggest_int("nb\_fc\_features", 1, 512)
    # modes = trial.suggest_categorical("modes", ["c", "cd", "cn", "cdn"])
    in_channels1 = trial.suggest_int("in\_channels1", 1, 180)
    in_channels2 = trial.suggest_int("in\_channels2", 1, 360)
    # in_channels3 = trial.suggest_int("in\_channels3", 1, 720)
    # in_channels4 = trial.suggest_int("in\_channels4", 1, 1440)
    
    # Compute the number of input channels given the modes
    # nb_input_channels = 0
    # if "c" in modes:
    #     nb_input_channels += 3
    # if "n" in modes:
    #     nb_input_channels += 3
    # if "d" in modes:
    #     nb_input_channels += 1
    
    
    # Load learning parameters
    LEARNING_PARAMS = params.learning.LEARNING
    
    # Update the learning parameters
    LEARNING_PARAMS.update({"learning_rate": lr,
                            # "weight_decay": wd,
                            "batch_size": batch_size})
    
    # Load network parameters
    NET_PARAMS = params.learning.NET_PARAMS2
    
    # Update the network parameters
    NET_PARAMS.update({
        # "img_channels": nb_input_channels,
        "in_channels1": in_channels1,
        "in_channels2": in_channels2,
        # "in_channels3": in_channels3,
        # "in_channels4": in_channels4,
        "num_fc_features": nb_fc_features})
    
    
    # Set the path to the dataset
    DATASET = "datasets/dataset_multimodal_siamese_png/"

    # Create a Dataset for training
    train_set = TraversabilityDataset(
        traversal_costs_file=DATASET+"traversal_costs_train.csv",
        images_directory=DATASET+"images_train",
        modes=params.learning.MODES,
        transform_image=transforms.train_transform,
        transform_depth=transforms.transform_depth,
        transform_normal=transforms.transform_normal
    )

    # Create a Dataset for validation
    val_set = TraversabilityDataset(
        traversal_costs_file=DATASET+"traversal_costs_train.csv",
        images_directory=DATASET+"images_train",
        modes=params.learning.MODES,
        transform_image=transforms.test_transform,
        transform_depth=transforms.transform_depth,
        transform_normal=transforms.transform_normal
    )

    # Create a Dataset for testin
    test_set = TraversabilityDataset(
        traversal_costs_file=DATASET+"traversal_costs_test.csv",
        images_directory=DATASET+"images_test",
        modes=params.learning.MODES,
        transform_image=transforms.test_transform,
        transform_depth=transforms.transform_depth,
        transform_normal=transforms.transform_normal
    )

    # Set the train dataset size
    train_size = params.learning.TRAIN_SIZE/(1-params.learning.TEST_SIZE)

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
        pin_memory=True,  # Increase the transferring speed of the data to the GPU
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
    # model = ResNet18Velocity(**params.learning.NET_PARAMS).to(device=device)   
    model = TraversabilityNet(**NET_PARAMS).to(device=device)
    
    # Define the loss function (combines nn.LogSoftmax() and nn.NLLLoss())
    criterion_classification = nn.CrossEntropyLoss()
    
    # Loss function to compare the expected traversal cost over the bins
    # and the ground truth traversal cost
    criterion_regression = nn.MSELoss()
    
    # Load the bins midpoints
    bins_midpoints = np.load(DATASET+"bins_midpoints.npy")
    bins_midpoints = torch.tensor(bins_midpoints[:, None],
                                  dtype=torch.float32,
                                  device=device)
    
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_PARAMS["learning_rate"],
                          momentum=LEARNING_PARAMS["momentum"],
                          weight_decay=LEARNING_PARAMS["weight_decay"])
    
    # Create tensors to store the loss and accuracy values
    loss_values = torch.zeros(2, LEARNING_PARAMS["nb_epochs"])
    accuracy_values = torch.zeros(2, LEARNING_PARAMS["nb_epochs"])
    regression_loss_values = torch.zeros(2, LEARNING_PARAMS["nb_epochs"])
    
    # Define the number of epochs to wait before stopping the training
    # if the validation loss does not improve
    # PATIENCE = 10
    
    # Train the model for multiple epochs
    best_val_regression_loss = np.inf
    best_epoch = 0
    
    # Loop over the epochs
    for epoch in range(LEARNING_PARAMS["nb_epochs"]):

        # Training
        train_loss, train_accuracy, train_regression_loss =\
            train(model,
                  device,
                  train_loader,
                  optimizer,
                  criterion_classification,
                  criterion_regression,
                  bins_midpoints,
                  epoch)

        # Validation
        val_loss, val_accuracy, val_regression_loss =\
            validate(model,
                     device,
                     val_loader,
                     criterion_classification,
                     criterion_regression,
                     bins_midpoints,
                     epoch)

        # print("Train accuracy: ", train_accuracy)
        # print("Validation accuracy: ", val_accuracy)
        # print("Train regression loss: ", train_regression_loss)
        # print("Validation regression loss: ", val_regression_loss)

        # Store the computed losses
        loss_values[0, epoch] = train_loss
        loss_values[1, epoch] = val_loss
        # Store the computed accuracies
        accuracy_values[0, epoch] = train_accuracy
        accuracy_values[1, epoch] = val_accuracy
        # Store the computed regression losses
        regression_loss_values[0, epoch] = train_regression_loss
        regression_loss_values[1, epoch] = val_regression_loss
        
        
        # Early stopping based on validation accuracy: stop the training if
        # the accuracy has not improved for the last 5 epochs
        if val_regression_loss < best_val_regression_loss:
            best_val_regression_loss = val_regression_loss
            best_epoch = epoch
        
        # elif epoch - best_epoch >= PATIENCE:
        #     print(f'Early stopping at epoch {epoch}')
        #     break
    
    # Set the uncertainty function
    uncertainty_function = uncertainty.functions.shannon_entropy

    # Test the model
    _,\
    test_accuracy,\
    test_regression_loss,\
    test_regression_losses,\
    uncertainties = test(model,
                         device,
                         test_loader,
                         criterion_classification,
                         criterion_regression,
                         bins_midpoints,
                         uncertainty_function)

    # Compute the test losses after successive removal of the samples
    # with the highest loss and uncertainty
    test_losses_loss, test_losses_uncertainty =\
        uncertainty.evaluation.uncertainty_relevance(
            model,
            device,
            criterion_classification,
            criterion_regression,
            bins_midpoints,
            uncertainty_function,
            test,
            test_set,
            test_regression_loss,
            test_regression_losses,
            uncertainties)
    
    
    # Get the absolute path of the current directory
    directory = os.path.abspath(os.getcwd())
    
    # Set the path to the results directory
    results_directory = directory +\
                        "/src/models_development/multimodal_velocity/logs_optuna2/_" +\
                        str(trial.number)
    
    # Get the learning parameters table
    params_table = parameters_table(dataset=DATASET,
                                    learning_params=LEARNING_PARAMS)

    # Generate the log directory
    generate_log(results_directory=results_directory,
                 test_regression_loss=test_regression_loss,
                 test_accuracy=test_accuracy,
                 parameters_table=params_table,
                 model=model,
                 regression_loss_values=regression_loss_values,
                 accuracy_values=accuracy_values,
                 test_losses_loss=test_losses_loss,
                 test_losses_uncertainty=test_losses_uncertainty)
    
    # Close all the previously opened figures
    plt.close("all")
    
    return best_val_regression_loss


# Set the number of trials
NB_TRIALS = 50

# Ignore warnings
warnings.filterwarnings("ignore")

# Samplers: GridSampler, RandomSampler, TPESampler, CmaEsSampler,
# PartialFixedSampler, NSGAIISampler, QMCSampler
# Pruners: MedianPruner, NopPruner, PatientPruner, PercentilePruner,
# SuccessiveHalvingPruner, HyperbandPruner, ThresholdPruner
study = optuna.load_study(
    storage="sqlite:///src/models_development/multimodal_velocity/logs_optuna2/optuna.db",
    study_name="multimodal_velocity",
    )
# study = optuna.create_study(
#     direction="minimize",
#     sampler=optuna.samplers.TPESampler(),
#     # pruner=optuna.pruners.MedianPruner()
#     storage="sqlite:///src/models_development/multimodal_velocity/logs_optuna2/optuna.db",
#     study_name="multimodal_velocity",
#     )
study.optimize(objective, n_trials=NB_TRIALS)
