""" Dataset creation parameters """


#################################
## Dataset creation parameters ##
#################################

# Linear velocity threshold from which the robot is considered to be moving
LINEAR_VELOCITY_THR = 0.05  # [m/s]

# Maximum number of rectangles to be detected in an image
NB_RECTANGLES_MAX = 3

# Distance the robot travels within a patch
PATCH_DISTANCE = 0.5  # [m]

# Threshold to filter tilted patches
PATCH_ANGLE_THR = 0.2  # [rad]

# Ratio between the width and the height of a rectangle
RECTANGLE_RATIO = 3

# Time during which the future trajectory is taken into account
T = 10  # [s]


####################
## Traversal cost ##
####################

# Number of levels for the discrete wavelet transform
NB_LEVELS = 2

# De-noising threshold for roll rate, pitch rate and vertical acceleration
# signals
DENOISE_THR = 0.005  # [rad/s | m/s^2]

# Wavelet name
WAVELET = "db3"

# Minimum length for a signal (depends on the wavelet and the number of levels)
SIGNAL_MIN_LENGTH = 20

# Padding mode for the signals that are too short
PADDING_MODE = "symmetric"

# Number of signals to characterize a patch
NB_SIGNALS = 3

# Number of features to extract from a signal
NB_FEATURES = (NB_LEVELS + 1)*NB_SIGNALS

# Number of bins to digitized the traversal cost
NB_BINS = 10

# Set the binning strategy
BINNING_STRATEGY = "kmeans"
