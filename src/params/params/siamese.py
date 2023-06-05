import numpy as np
from datetime import datetime

# Import custom packages
import traversalcost.features


#########################################################
## Dataset creation parameters for the Siamese network ##
#########################################################

# Terrain classes ordered from the most to the least traversable
ORDERED_TERRAIN_CLASSES = ["road", "sand", "grass"]

# List of linear velocities sent to the robot when recording IMU signals to
# design a traversal cost
LINEAR_VELOCITIES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Describe the features to be extracted from the IMU signals
# (if the function takes parameters, default values can be overwritten by
# specifying them in dictionaries)
# (the output of the function must be a numpy array of shape (n,) or a list
# of length n, n being the number of features)
FEATURES = {"function": traversalcost.features.dwt_variances,
            "params_roll_rate": {},
            "params_pitch_rate": {},
            "params_vertical_acceleration": {}}


#############################################
## Parameters to train the Siamese network ##
#############################################

# Path to the dataset to be used (relative to the notebook path)
DATASET = "../datasets/dataset_40Hz_dwt_hard/"

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

LEARNING = {"batch_size": 8,
            "nb_epochs": 200,
            "margin": 0.15,
            "learning_rate": 0.005,
            "weight_decay": 0.,
            # "weight_decay": 0.001,
            "momentum": 0.9}

# # Batch size
# BATCH_SIZE = 8

# # Number of epochs
# NB_EPOCHS = 100

# # The margin used in the loss to prevent the network from learning a
# # trivial solution
# MARGIN = 0.5

# # Optimizer parameters
# LEARNING_RATE = 0.001
# WEIGHT_DECAY = 0.001
# MOMENTUM = 0.9


#######################################
## Saving the weights of the network ##
#######################################

# The name of the file in which the weights of the network will be saved
PARAMS_FILE = "siamese.params"


############################
## Output of the training ##
############################

# The name of the directory in which the logs will be saved
LOG_DIR = None
