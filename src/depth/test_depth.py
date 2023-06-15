"""
A script to test the functions of the depth package.
"""

# Import modules and packages
import cv2
import os

# Import custom packages
import params.robot
from depth.utils import Depth


# Get the absolute path of the current directory
directory = os.path.abspath(os.getcwd())

# Set the path to the example depth image
depth_image_path = directory + "/src/depth/depth_image.tiff"

# Load the depth image
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

# Create a Depth object from a depth image
depth = Depth(depth_image)

# Display the depth image
depth.display_depth()

# Set the parameters for the bilateral filter
bilateral_filter = {"d": 5,
                    "sigmaColor": 0.5,
                    "sigmaSpace": 2}

# Compute the surface normals
depth.compute_normal(K=params.robot.K,
                     bilateral_filter=bilateral_filter,
                     gradient_threshold=10)

# Display the normal image
depth.display_normal()

cv2.waitKey(0)
