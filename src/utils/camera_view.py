"""
Display images captured and stored in a bagfile
"""

# ROS - Python librairies
import rosbag

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

# Python librairies
import cv2

# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_path.bag"
# FILE = "bagfiles/sample_bag.bag"
# Name of the image topic
# IMAGE_TOPIC = "/zed_node/depth/depth_registered"
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"

# Initialize the bridge between ROS and OpenCV images
bridge = cv_bridge.CvBridge()

# Open the bag file
bag = rosbag.Bag(FILE)

# Read the bag file
for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC]):
    # print(msg.type)
    # Convert the ROS Image type to a numpy ndarray
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    # Display the image
    cv2.imshow("Preview", image)
    # Wait for 300 ms (for a key press) before automatically destroying the
    # current window
    cv2.waitKey(300)

# Close the bag file
bag.close()
