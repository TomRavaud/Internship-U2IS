"""
Dataset creation: rectangular regions of the future robot path are
extracted from the image and associated with a traversal cost computed
from statistics on IMU measurements
"""

# ROS - Python librairies
import rosbag  # Read bag files
import tf.transformations
import cv_bridge
import rospy

# Python librairies
import csv
# Provide operating system functionalities
import os
# Provide access to some variables used or maintained by the Python
# interpreter and to functions that interact strongly with it
import sys
# To create progress bars
from tqdm import tqdm
# Image processing (Python Imaging Library)
from PIL import Image
import time
import numpy as np
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

def inverse_transform_matrix(HTM):
    """Compute the inverse matrix of an homogeneous transform matrix

    Args:
        HTM (ndarray (4, 4)): a transform matrix

    Returns:
        ndarray (4, 4): inverse matrix of the transform matrix
    """
    # Extract the rotation matrix and the translation vector
    R = HTM[:3, :3]
    t = HTM[:3, 3]
    
    # Compute the inverse transform
    HTM_inverse = np.zeros_like(HTM)
    HTM_inverse[:3, :3] = np.transpose(R)
    HTM_inverse[:3, 3] = -np.dot(np.transpose(R), t)
    HTM_inverse[3, 3] = 1
    
    return HTM_inverse

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
        cv2.circle(image, tuple(point), radius=4,
                   color=color, thickness=-1)
        
    return image

def draw_quadrilateral(image, corners, color=(0, 0, 255)):
    """Draw a quadrilateral from its four corners
    Args:
        image (cv::Mat): an OpenCV image
        corners (ndarray (4, 2)): ordered array of the corners
        color (tuple, optional): color of the quadrilateral. Defaults to (0, 0, 255).
    Returns:
        cv::Mat: the modified image
    """
    corners = corners.astype(np.int32)
    
    # Link the corners with 4 lines
    image = cv2.line(image, tuple(corners[0]), tuple(corners[1]), color, 2)
    image = cv2.line(image, tuple(corners[1]), tuple(corners[2]), color, 2)
    image = cv2.line(image, tuple(corners[2]), tuple(corners[3]), color, 2)
    image = cv2.line(image, tuple(corners[3]), tuple(corners[0]), color, 2)
    
    return image

def traversal_cost(roll_angle_values,
                   pitch_angle_values,
                   roll_velocity_values,
                   pitch_velocity_values):
    """Compute a traversal cost from IMU measurements

    Args:
        roll_angle_values (list): roll angle values
        pitch_angle_values (list): pitch angle values
        roll_velocity_values (list): roll velocity values
        pitch_velocity_values (list): pitch velocity values

    Returns:
        float: traversal score computed from statistics on IMU measurements
    """
    # Compute the mean of the roll and pitch values
    roll_angle_mean = np.mean(roll_angle_values)
    pitch_angle_mean = np.mean(pitch_angle_values)
    
    # Compute the variance of the roll and pitch velocities
    roll_velocity_variance = np.var(roll_velocity_values)
    pitch_velocity_variance = np.var(pitch_velocity_values)
    
    return pitch_velocity_variance


# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_road.bag"  # TODO: here

# Topics name
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"
ODOM_TOPIC = "/odometry/filtered"
IMU_TOPIC = "imu/data"  # TODO: here

# Time during which the future trajectory is taken into account
T = 3  # seconds

# Time step for trajectory sampling
dt = 0.5  # seconds

# Odometry measurements frequency
f = 50  # Hz

# Number of odometry measurements within dt second(s)
div = np.int32(f*dt)

# Downsampling ratio : if equal to 5, 1 observation out of 5 will be saved
# DOWNSAMPLING_RATIO = 2

# Distance between the left and the right wheels of the robot
L = 0.67

# (Constant) Transform matrix from the IMU frame to the camera frame
alpha = -0.197  # Camera tilt (approx -11.3 degrees)
ROBOT_TO_CAM = np.array([[0, np.sin(alpha), np.cos(alpha), 0.084],
                         [-1, 0, 0, 0.060],
                         [0, -np.cos(alpha), np.sin(alpha), 0.774],
                         [0, 0, 0, 1]])

# Inverse the transform
CAM_TO_ROBOT = inverse_transform_matrix(ROBOT_TO_CAM)

# (Constant) Internal calibration matrix (approx focal length)
# TODO: here
# K = np.array([[700, 0, 320],
#               [0, 700, 180],
#               [0, 0, 1]])
K = np.array([[700, 0, 640],
              [0, 700, 360],
              [0, 0, 1]])

# Initialize the bridge between ROS and OpenCV images
bridge = cv_bridge.CvBridge()


print("Reading file : " + FILE)

# Open the bag file
bag = rosbag.Bag(FILE)

# Get the name of the file (without parent directories)
bag_name = os.path.basename(bag.filename)

# Create a new directory to store the dataset
# Get the absolute path of the current directory
directory = os.path.abspath(os.getcwd())

# Set the name of the directory which will store the dataset
# TODO: here
results_dir = directory + "/datasets/dataset_test"
# results_dir = directory + "/datasets/dataset_all"
# results_dir = directory + "/datasets/dataset_" + bag_name[:-4]

try:  # A new directory is created if it does not exist yet
    os.mkdir(results_dir)
    print(results_dir + " folder created\n")
    
except OSError:  # Display a message if it already exists and quit
    print("Existing directory " + results_dir)
    print("Aborting to avoid overwriting data\n")
    # sys.exit(1)  # Stop the execution of the script
    pass


# Create a sub-directory to store images and their associated timestamp
topic_name = IMAGE_TOPIC[1:].replace('/', '_')
topic_dir = results_dir + "/" + topic_name

# Create a directory if it does not exist yet
try:
    os.mkdir(topic_dir)
    print(topic_dir + " folder created\n")
except OSError:
    pass

# Create a csv file to store the traversal scores
csv_name = results_dir + "/traversal_costs.csv"

# Open the file in the write mode
file_costs = open(csv_name, "a")   #TODO: here

# Create a csv writer
file_costs_writer = csv.writer(file_costs, delimiter=",")

# Write the first row (columns title)
#TODO: here
# headers = ["image_id", "traversal_cost"]
# file_costs_writer.writerow(headers)


print("Processing images")

# Variable to keep the index of the currently processed image
index_image = 0  #TODO: here


# Go through the image topic
for _, msg_image, t_image in tqdm(bag.read_messages(topics=[IMAGE_TOPIC]),
                                  total=bag.get_message_count(IMAGE_TOPIC)):

    # Convert the current ROS image to the OpenCV type
    image = bridge.imgmsg_to_cv2(msg_image, desired_encoding="passthrough")
    
    # Get the first odometry message received after the image
    _, msg_odom, t_odom = next(iter(bag.read_messages(
        topics=[ODOM_TOPIC],
        start_time=t_image)))

    # Compute the transform matrix between the world and the robot
    WORLD_TO_ROBOT = pose_to_transform_matrix(msg_odom.pose.pose)

    # Compute the inverse transform
    ROBOT_TO_WORLD = inverse_transform_matrix(WORLD_TO_ROBOT)

    # Initialize an array to store the previous front wheels
    # coordinates in the image
    points_image_old = 1e5*np.ones((2, 2))
    
    # Define a variable to store the previous timestamp
    t_odom_old = None

    
    # Read the odometry measurement for T second(s)
    for i, (_, msg_odom, t_odom) in enumerate(bag.read_messages(
        topics=[ODOM_TOPIC],
        start_time=t_odom,
        end_time=t_odom+rospy.Duration(T))):
        
        # Keep only an odometry measurement every dt second(s)
        if i % div == 0:
            
            # Store the 2D coordinates of the robot position in the world frame
            point_world = np.array([[msg_odom.pose.pose.position.x,
                                     msg_odom.pose.pose.position.y,
                                     msg_odom.pose.pose.position.z]])

            # Make the orientation quaternion a numpy array
            q = np.array([msg_odom.pose.pose.orientation.x,
                          msg_odom.pose.pose.orientation.y,
                          msg_odom.pose.pose.orientation.z,
                          msg_odom.pose.pose.orientation.w])

            # Convert the quaternion into Euler angles
            theta = tf.transformations.euler_from_quaternion(q)[2]
            
            # Create arrays to store left and write front wheels positions
            point_left_world = np.copy(point_world)
            point_right_world = np.copy(point_world)

            # Compute the distances between the wheels and the robot's origin
            delta_X = L*np.sin(theta)/2
            delta_Y = L*np.cos(theta)/2

            # Compute the positions of the outer points of the two
            # front wheels
            point_left_world[:, 0] -= delta_X
            point_left_world[:, 1] += delta_Y
            point_right_world[:, 0] += delta_X
            point_right_world[:, 1] -= delta_Y

            # Gather front wheels outer points coordinates in a single array
            points_world = np.concatenate([point_left_world,
                                           point_right_world])

            # Compute the points coordinates in the robot frame
            points_robot = apply_rigid_motion(points_world,
                                              ROBOT_TO_WORLD)

            # Compute the points coordinates in the camera frame
            points_camera = apply_rigid_motion(points_robot,
                                               CAM_TO_ROBOT)

            # Compute the points coordinates in the image plan
            points_image = camera_frame_to_image(points_camera,
                                                 K)
            
            # Draw the points on the image
            image = draw_points(image, points_image)
            
            # Compute the maximum and minimum coordinates on the y axis of the
            # image plan
            max_y = np.int32(np.max(points_image_old, axis=0)[1])
            min_y = np.int32(np.min(points_image, axis=0)[1])
            
            
            # Process the points only if they are visible in the image
            if max_y < image.shape[0] and min_y > 0:
                
                # Create lists to store all the roll, pitch angles and
                # velocities measurement within the dt second(s) interval
                roll_angle_values = []
                pitch_angle_values = []
                roll_velocity_values = []
                pitch_velocity_values = []

                # Read the IMU measurements within the dt second(s) interval
                for _, msg_imu, t_imu in bag.read_messages(
                    topics=[IMU_TOPIC],
                    start_time=t_odom_old,
                    end_time=t_odom):

                    # Append angles and angular velocities to the previously
                    # created lists
                    roll_angle_values.append(msg_imu.orientation.x)
                    pitch_angle_values.append(msg_imu.orientation.y)
                    roll_velocity_values.append(msg_imu.angular_velocity.x)
                    pitch_velocity_values.append(msg_imu.angular_velocity.y)

                # Compute a traversal cost based on IMU measurements
                # statistics
                cost = traversal_cost(roll_angle_values,
                                      pitch_angle_values,
                                      roll_velocity_values,
                                      pitch_velocity_values)

                # print("Traversal cost: ", cost)
                # print("\n")
                
                # Compute max and min coordinates of the points in the image
                # along the x axis
                min_x = np.int32(np.min([points_image_old[:, 0],
                                     points_image[:, 0]]))
                max_x = np.int32(np.max([points_image_old[:, 0],
                                     points_image[:, 0]]))
                
                # Draw a rectangle in the image to visualize the region
                # of interest
                image = draw_quadrilateral(image,
                                           np.array([[min_x, max_y],
                                                     [min_x, min_y],
                                                     [max_x, min_y],
                                                     [max_x, max_y]]),
                                           color=(255, 0, 0))
                
                # Extract the rectangular region of interest from
                # the original image
                image_to_save = image[min_y:max_y, min_x:max_x]
                
                
                # Keep only rectangles which surface is greater than a
                # given value
                if image_to_save.shape[0]*image_to_save.shape[1] >= 10000:
                    print("Ratio W/H: ", image_to_save.shape[1]/image_to_save.shape[0])
                    print(image_to_save.shape)
                    
                    # print("Image ", index_image, " surface : ",\
                    # image_to_save.shape[0]*image_to_save.shape[1])

                    # Convert the image from BGR to RGB
                    image_to_save = cv2.cvtColor(image_to_save,
                                                 cv2.COLOR_BGR2RGB)
                    
                    # Make a PIL image
                    image_to_save = Image.fromarray(image_to_save)

                    # Give the image a name
                    image_name = f"{index_image:05d}.png"

                    # Save the image in the correct directory
                    # image_to_save.save(topic_dir + "/" + image_name, "PNG")  #TODO: here
                    
                    # Add the image index and the associated score in the csv file
                    # file_costs_writer.writerow([image_name, cost])  #TODO: here
                    
                    index_image += 1

                    # Display the image
                    cv2.imshow("Image", image)
                    cv2.waitKey()
            
            # Update front wheels outer points image coordinates and timestamp
            points_image_old = points_image
            t_odom_old = t_odom
                

# Close the bag file
bag.close()

# Close the csv file
file_costs.close()
