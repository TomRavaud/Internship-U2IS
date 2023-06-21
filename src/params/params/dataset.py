""" Dataset creation parameters """

import torch

# Import custom packages
import traversalcost.features


#################################
## Dataset creation parameters ##
#################################

# Upper bound of the number of images to be extracted from the bag files
NB_IMAGES_MAX = 100000

# Number of messages threshold before skipping a rosbag file
NB_MESSAGES_THR = 0.08

# Linear velocity threshold from which the robot is considered to be moving
LINEAR_VELOCITY_THR = 0.05  # [m/s]

# Maximum number of rectangles to be detected in an image
NB_RECTANGLES_MAX = 3

# Distance the robot travels within a patch
PATCH_DISTANCE = 0.5  # [m]

# Threshold to filter tilted patches
PATCH_ANGLE_THR = 1  # [rad]

# Ratio between the width and the height of a rectangle
RECTANGLE_RATIO = 3

# Time during which the future trajectory is taken into account
T = 10  # [s]

# Time interval to look for the depth image corresponding to the current rgb
# image
TIME_DELTA = 0.05  # [s]


##########################################
## Features extraction from IMU signals ##
##########################################

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


##################################################
## Traversal cost computation from the features ##
##################################################

#-------------------#
#  Siamese Network  #
#-------------------#

# Path to the parameters file
SIAMESE_PARAMS = "src/traversal_cost/siamese_network/logs/_2023-06-12-19-33-37/siamese.params"

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


###########################################################
## Depth image and surface normal computation parameters ##
###########################################################

# Set the parameters for the bilateral filter
# BILATERAL_FILTER = {"d": 5,
#                     "sigmaColor": 0.5,
#                     "sigmaSpace": 2}
BILATERAL_FILTER = None

# Threshold for the gradient magnitude
GRADIENT_THR = 8

# Set the depth range
DEPTH_RANGE = (0.7, 7)  # [m]

# Set the default normal vector to replace the invalid ones
DEFAULT_NORMAL = [0, 0, 1]


###########################
## Train and test splits ##
###########################

# Tell if the splits must be stratified or not
STRATIFY = False
