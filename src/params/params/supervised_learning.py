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
DATASET ="../datasets/dataset_big_DS_test/"

#Dictionnary with cost functions according to terrains used when creating dataframe
terrain_cost_small = {"grass": lambda x: 3*x,
               "road": lambda x: 0.5*x, 
               "sand": lambda x: 2*x}

terrain_cost= {"road_easy" : lambda x: x,
    "road_medium" : lambda x: 2*x,
    "forest_dirt_easy" : lambda x: 2.1*x,
    "dust" : lambda x: 2*x,
    "forest_leaves" : lambda x: 1.75*x,
    "forest_dirt_medium" : lambda x: 1.80 * x,
    "gravel_easy" : lambda x: 3*x,
    "grass_easy" : lambda x: 1.1*x,
    "grass_medium" : lambda x: 1.45*x,
    "gravel_medium" : lambda x: 1.95*x,
    "forest_leaves_branches" : lambda x: 1.55*x,
    "forest_dirt_stones_branches" : lambda x: 2.2*x}
    # "sand_hard",
    # "sand_medium", 
    

#Lenght of the test set
LENGHT_TEST_SET = 0.1

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Batch size
BATCH_SIZE = 5

# Number of epochs
NB_EPOCHS = 30

# Optimizer parameters
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
MOMENTUM = 0.4
