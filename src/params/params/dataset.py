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
