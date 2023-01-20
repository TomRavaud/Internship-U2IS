"""
Draw the robot path on a 2D map based on odometry measurements
"""

# ROS - Python librairies
import rosbag
import tf.transformations 
import cv_bridge

import numpy as np
import matplotlib.pyplot as plt
import cv2


def pose_to_transform_matrix(pose_msg):
    """Convert a Pose object to a numpy transform matrix
    Args:
        pose_msg (Pose): a ROS Pose message which contains
        a position vector and an orientation matrix
    Returns:
        ndarray (4, 4): the corresponding transform matrix
    """
    # Make the translation vector a numpy array
    T = np.array([pose_msg.position.x,
                  pose_msg.position.y,
                  pose_msg.position.z])
    
    # Make the quaternion a numpy array
    q = np.array([pose_msg.orientation.x,
                  pose_msg.orientation.y,
                  pose_msg.orientation.z,
                  pose_msg.orientation.w])
    
    # Form the transform matrix from the translation and the quaternion
    HTM = tf.transformations.quaternion_matrix(q)
    HTM[0:3, 3] = T
    
    return HTM

def apply_rigid_motion(points, HTM):
    """Give points' coordinates in a new frame obtained after rotating (R)
    and translating (T) the current one
    Args:
        points (ndarray (N, 3)): a set of points
        HTM (ndarray (4, 4)): a homogeneous transform matrix
    Returns:
        ndarray (N, 3): points new coordinates
    """
    # Number of points we want to move
    nb_points = np.shape(points)[0]
    
    # Use homogenous coordinates
    homogeneous_points = np.ones((nb_points, 4))
    homogeneous_points[:, :-1] = points
    
    # Compute points coordinates after the rigid motion
    points_new = np.dot(homogeneous_points, np.transpose(HTM[:3, :]))
    
    return points_new

def camera_frame_to_image(points, K):
    """Compute points coordinates in the image frame from their coordinates in
    the camera frame
    Args:
        points (ndarray (N, 3)): a set of points
        K (ndarray (3, 3)): the internal calibration matrix
    Returns:
        ndarray (N, 2): points image coordinates
    """
    # Project the points onto the image plan, the obtained coordinates are
    # defined up to a scaling factor
    points_projection = np.dot(points, np.transpose(K))
    
    # Get the points' coordinates in the image frame dividing by the third
    # coordinate
    points_image = points_projection[:, :2]/points_projection[:, 2][:, np.newaxis]
 
    return points_image

def draw_points(image, points, color=(0, 0, 255)):
    """Draw some points on an image
    Args:
        image (cv::Mat): an OpenCV image
        points (ndarray (N, 2)): a set of points
        color (tuple, optional): color of the points. Defaults to (0, 0, 255).
    Returns:
        cv::Mat: the modified image
    """
    # Convert points coordinates to int to correspond to pixel values
    points = points.astype(np.int32)
    
    # Draw a red circle on the image for point
    for point in points:
        cv2.circle(image, tuple(point), radius=3,
                   color=color, thickness=-1)
        
    return image


# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_road.bag"
# Name of the odometry topic
ODOM_TOPIC = "/odometry/filtered"
# Name of the image topic
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"


# Transform matrix from the IMU frame to the camera frame
alpha = 0.197  # Camera tilt (approx 11.3 degrees)
ROBOT_TO_CAM = np.array([[0, -1, 0, 0],
                         [-np.sin(alpha), 0, -np.cos(alpha), 0.6],
                         [np.cos(alpha), 0, -np.sin(alpha), 0],
                         [0, 0, 0, 1]])

# Internal calibration matrix (approx focal length)
K = np.array([[700, 0, 640],
              [0, 700, 360],
              [0, 0, 1]])

# Initialize the bridge between ROS and OpenCV images
bridge = cv_bridge.CvBridge()

# Open the bag file
bag = rosbag.Bag(FILE)

# Read the first image of the bagfile
_, msg_image, t_image = next(iter(bag.read_messages(topics=[IMAGE_TOPIC])))

# State equal True if the transform between the world and the robot has been
# computed
transform = False

# Number of recorded positions of the robot after the first
# image has been taken
nb_points = 2765

# Create an array to contain the successives coordinates of the
# robot in the world frame
points_world = np.zeros((2765, 3), dtype=np.float32)

# Index of the current position of the robot
i = 0

# Read the bag file (odometry topic)
for _, msg_odom, t_odom in bag.read_messages(topics=[ODOM_TOPIC]):
    
    # Detect the first odometry measurement after the image has been taken
    if t_odom >= t_image and transform == False:
        
        # Compute the transform matrix between the robot and the world
        ROBOT_TO_WORLD = pose_to_transform_matrix(msg_odom.pose.pose)
        
        # Extract the rotation matrix and the translation vector
        R = ROBOT_TO_WORLD[:3, :3]
        t = ROBOT_TO_WORLD[:3, 3]
        
        # Compute the inverse transform
        WORLD_TO_ROBOT = np.zeros_like(ROBOT_TO_WORLD)
        WORLD_TO_ROBOT[:3, :3] = np.transpose(R)
        WORLD_TO_ROBOT[:3, 3] = -np.dot(np.transpose(R), t)
        WORLD_TO_ROBOT[3, 3] = 1
        
        # Indicate the transform has been computed
        transform = True
    
    # For the odometry measurements that appeared after the image
    # has been taken
    if t_odom >= t_image:
        
        # Store the 2D coordinates of the robot position in the world frame
        points_world[i] = np.array([msg_odom.pose.pose.position.x,
                                    msg_odom.pose.pose.position.y,
                                    msg_odom.pose.pose.position.z])
        
        i += 1

# Close the bag file
bag.close()


# Get the coordinates of the successive robot positions
X = points_world[:, 0]
Y = points_world[:, 1]

# Display the evolution of the robot position in the world frame
plt.figure("Robot position")
plt.plot(X, Y, ".")
plt.plot(X[0], Y[0], "ro")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Evolution of the robot 2D position")
plt.show()


# Compute the points coordinates in the robot frame
points_robot = apply_rigid_motion(points_world,
                                  WORLD_TO_ROBOT)

# Compute the points coordinates in the camera frame
points_camera = apply_rigid_motion(points_robot,
                                   ROBOT_TO_CAM)

# Compute the points coordinates in the image plan
points_image = camera_frame_to_image(points_camera,
                                     K)

# Convert the ROS Image type to a numpy ndarray
image = bridge.imgmsg_to_cv2(msg_image, desired_encoding="bgr8")

# Draw the points on the image
image = draw_points(image, points_image)

# Display the image
cv2.imshow("Robot path", image)

# Wait for a key press to close the window
cv2.waitKey()
