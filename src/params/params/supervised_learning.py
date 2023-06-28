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

# Path of the dataset
DATASET = "../datasets/dataset_to_delete/"

# Set the cost functions used to generate traversal costs given the terrain
# type and the linear velocity of the robot
# terrain_cost = {"grass": lambda x: 3*x,
#                "road": lambda x: 0.5*x, 
#                "sand": lambda x: 2*x}
terrain_cost = {
    "road_easy" : lambda x: 2*x,
    "road_medium" : lambda x: 2*x + 1,
    "forest_dirt_easy" : lambda x: 2*x + 2,
    "dust" : lambda x: 2*x + 3,
    "forest_leaves" : lambda x: 2*x + 4,
    "forest_dirt_medium" : lambda x: 2*x + 5,
    "gravel_easy" : lambda x: 2*x + 6,
    "grass_easy" : lambda x: 2*x + 7,
    "grass_medium" : lambda x: 2*x + 8,
    "gravel_medium" : lambda x: 2*x + 9,
    "forest_leaves_branches" : lambda x: 2*x + 10,
    "forest_dirt_stones_branches" : lambda x: 2*x + 11,
    }

# The name of the file in which the weights of the network will be saved
PARAMS_FILE = "supervised_learning.params"

# Set the learning parameters
LEARNING = {"batch_size": 10,
            "nb_epochs": 50,
            "learning_rate": 0.0005,
            "weight_decay": 1e-5,
            "momentum": 0.9}

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Set the path of the log directory
LOG_DIR = None
