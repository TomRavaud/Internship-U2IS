import numpy as np
from datetime import datetime

# Import custom packages
import traversalcost.features


#########################################################
## Dataset creation parameters for the Supervised network ##
#########################################################



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

#Input size of the feature used in the neuronal network
INPUT_FEATURE_SIZE = 78

#Link to the Dataset
DATASET = "/home/student/Desktop/Stage ENSTA/Internship-U2IS/src/traversal_cost/datasets/dataset_small_DS_test/"


#Dictionnary with cost functions according to terrains used when creating dataframe
terrain_cost = {"grass": lambda x: 3*x,
               "road": lambda x: 0.5*x, 
               "sand": lambda x: 2*x}

terrain_cost_small = {
    "road_easy" : lambda x: x,
    "road_medium" : lambda x: 2*x,
    "dust" : lambda x: 3*x,
    "forest_leaves" : lambda x: 4*x,
    "forest_dirt_easy" : lambda x: 5*x,
    "forest_dirt_medium" : lambda x: 6 * x,
    "forest_leaves_branches" : lambda x: 7*x,
    "forest_dirt_stones_branches" : lambda x: 7.5*x,
    "gravel_easy" : lambda x: 3.5*x,
    "grass_easy" : lambda x: 1.5*x,
    "grass_medium" : lambda x: 2.5*x,
    "gravel_medium" : lambda x: 4.5*x,
    }
    # "sand_hard",
    # "sand_medium", 
    
# The name of the file in which the weights of the network will be saved
PARAMS_FILE = "supervised_learning.params"

LEARNING = {"batch_size": 64,
            "nb_epochs": 10,
            "margin": 0.15,
            "learning_rate": 0.005,
            "weight_decay": 0.0001,
            "momentum": 0.9}

#Lenght of the test set
LENGHT_TEST_SET = 0.25

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Batch size
BATCH_SIZE = 1

# Number of epochs
NB_EPOCHS = 15

# Optimizer parameters
LEARNING_RATE = 0.0022
WEIGHT_DECAY = 7.41e-07 #prevents from overtifiting
MOMENTUM = 0.9
