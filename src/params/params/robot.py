""" Robot parameters """

import numpy as np
import utils.frames as frames


###########################################
## Robot characteristics and accessories ##
###########################################

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


################
## ROS topics ##
################

# Topics' name
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"
DEPTH_TOPIC = "/zed_node/depth/depth_registered"
ODOM_TOPIC = "/odometry/filtered/local"
WHEELS_ODOM_TOPIC = "/husky_velocity_controller/odom"
VISUAL_ODOM_TOPIC = "/zed_node/odom"
IMU_TOPIC = "/imu/data"


#############################
## Sensors characteristics ##
#############################

# (Constant) Internal calibration matrix (approx focal length)
K = np.array([[534, 0, 634],
              [0, 534, 363],
              [0, 0, 1]])

# Odometry measurements frequency
ODOM_SAMPLE_RATE = 50  # Hz

# IMU measurements frequency
IMU_SAMPLE_RATE = 100  # Hz

# Image frequency
CAMERA_SAMPLE_RATE = 3  # Hz

# Depth image frequency
DEPTH_SAMPLE_RATE = 3  # Hz
