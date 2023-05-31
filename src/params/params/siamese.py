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

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Batch size
BATCH_SIZE = 3

# Number of epochs
NB_EPOCHS = 10
