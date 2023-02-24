"""
Define the parameters needed
"""
# Python libraries
import numpy as np

# Custom modules
import frames


## ROS ##

# Topics name
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"
ODOM_TOPIC = "/odometry/filtered"
IMU_TOPIC = "imu/data"

# Topics publishing rate
IMU_SAMPLE_RATE = 43  # Hz
ODOM_SAMPLE_RATE = 50  # Hz


## Robot and camera ##

# Distance between the left and the right wheels of the robot
L = 0.67

# (Constant) Transform matrix from the IMU frame to the camera frame
alpha = -0.197  # Camera tilt (approx -11.3 degrees)
ROBOT_TO_CAM = np.array([[0, np.sin(alpha), np.cos(alpha), 0.084],
                         [-1, 0, 0, 0.060],
                         [0, -np.cos(alpha), np.sin(alpha), 0.774],
                         [0, 0, 0, 1]])
# Inverse the transform
CAM_TO_ROBOT = frames.inverse_transform_matrix(ROBOT_TO_CAM)

# (Constant) Internal calibration matrix (approx focal length)
K = np.array([[534, 0, 634],
              [0, 534, 363],
              [0, 0, 1]])


## Trajectory sampling ##


## Learning ##
