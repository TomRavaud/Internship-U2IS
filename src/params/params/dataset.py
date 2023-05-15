""" Dataset creation parameters """


#################################
## Dataset creation parameters ##
#################################

# Linear velocity threshold from which the robot is considered to be moving
linear_velocity_threshold = 0.05  # [m/s]

# Maximum number of rectangles to be detected in an image
nb_rectangles_max = 3

# Distance the robot travels within a patch
patch_distance = 0.5  # [m]

# Threshold to filter tilted patches
patch_angle_threshold = 0.2  # [rad]

# Ratio between the width and the height of a rectangle
rectangle_ratio = 3

# Time during which the future trajectory is taken into account
T = 10  # [s]


####################
## Traversal cost ##
####################

# Number of levels for the discrete wavelet transform
nb_levels = 2

# De-noising threshold for roll rate, pitch rate and vertical acceleration
# signals
denoise_threshold = 0.005  # [rad/s | m/s^2]

# Wavelet name
wavelet = "db3"

# Minimum length for a signal (depends on the wavelet and the number of levels)
signal_min_length = 20

# Padding mode for the signals that are too short
padding_mode = "symmetric"

# Number of signals to characterize a patch
nb_signals = 3

# Number of features to extract from a signal
nb_features = (nb_levels + 1)*nb_signals

# Number of bins to digitized the traversal cost
nb_bins = 10

# Set the binning strategy
binning_strategy = "kmeans"
