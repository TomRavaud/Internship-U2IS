import numpy as np

# Import custom packages
import traversalcost.time_features


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
FEATURES = {"function": traversalcost.time_features.variance,
            "params_roll_rate": {},
            "params_pitch_rate": {},
            "params_vertical_acceleration": {}}


#############################################
## Parameters to train the Siamese network ##
#############################################

# Path to the dataset to be used (relative to the notebook path)
DATASET = "../datasets/dataset_40Hz/"

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Batch size
BATCH_SIZE = 3

# Number of epochs
NB_EPOCHS = 10

# The margin used in the loss to avoid the network to learn a trivial solution
MARGIN = 2

# Optimizer parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
MOMENTUM = 0.9
