#########################################################
## Dataset creation parameters for the Siamese network ##
#########################################################

# Terrain classes ordered from the most to the least traversable
ORDERED_TERRAIN_CLASSES = ["road", "sand", "grass"]

# List of linear velocities sent to the robot when recording IMU signals to
# design a traversal cost
LINEAR_VELOCITIES = [0.2, 0.4, 0.6, 0.8, 1.0]


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
