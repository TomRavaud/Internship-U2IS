"""
This script helps in reading depth images from rosbag files and saving them
as a tiff files.
"""

import rosbag
import numpy as np
import cv2
import cv_bridge

# Import custom modules and packages
import params.robot


# Create a bridge object to convert ROS to OpenCV images
bridge = cv_bridge.CvBridge()

# Open the bag file
bag = rosbag.Bag("bagfiles/raw_bagfiles/depth/tom_full1.bag")

# Loop through the messages in the bag file, read the depth image topic
for topic, msg, t in bag.read_messages(topics=[params.robot.DEPTH_TOPIC]):
    
    # Convert the ROS Image to an opencv image
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    # Save the current depth image as a tiff file
    cv2.imwrite("src/depth/depth_image.tiff", depth_image)
    break
