import numpy as np
from datetime import datetime

# Import custom packages
import traversalcost.features


#########################################################
## Dataset creation parameters for the Siamese network ##
#########################################################

# Terrain classes ordered from the most to the least traversable
ORDERED_TERRAIN_CLASSES =[
    "road_easy",
    "road_medium",
    "forest_dirt_easy",
    "dust",
    "forest_leaves",
    "forest_dirt_medium",
    "gravel_easy",
    "grass_easy",
    "grass_medium",
    "gravel_medium",
    "forest_leaves_branches",
    "forest_dirt_stones_branches",
    # "sand_hard",
    # "sand_medium",
    ]

# List of linear velocities sent to the robot when recording IMU signals to
# design a traversal cost
LINEAR_VELOCITIES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Describe the features to be extracted from the IMU signals
# (if the function takes parameters, default values can be overwritten by
# specifying them in dictionaries)
# (the output of the function must be a numpy array of shape (n,) or a list
# of length n, n being the number of features)
params = {}
FEATURES = {"function": traversalcost.features.wrapped_signal_fft,
            "params_roll_rate": params,
            "params_pitch_rate": params,
            "params_vertical_acceleration": params}


#############################################
## Parameters to train the Siamese network ##
#############################################

# Path to the dataset to be used (relative to the notebook path)
DATASET = "../datasets/dataset_200Hz_wrap_fft/"

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

LEARNING = {"batch_size": 64,
            "nb_epochs": 100,
            "margin": 0.15,
            "learning_rate": 0.005,
            "weight_decay": 0.0001,
            "momentum": 0.9}


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
