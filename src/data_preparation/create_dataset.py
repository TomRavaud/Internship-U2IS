"""
Script to convert a bagfile containing images, IMU and odometry data into
a self-supervised dataset for terrain traversability analysis
"""

# ROS - Python librairies
import rosbag  # Read bag files
import rospy
from cv_bridge import CvBridge # To convert images types
from tf.transformations import euler_from_quaternion



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


# Measure time of the dataset creation procedure
start = time.time()

# Define some constants
# Topics name
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"
ODOM_TOPIC = "/odometry/filtered"
IMU_TOPIC = "/imu/data"

# Downsampling ratio : if equal to 5, 1 observation out of 5 will be saved
DOWNSAMPLING_RATIO = 2

# To deal with differences in publishing times on topics
# EPSILON_T corresponds to the time difference between them we allow
EPSILON_T = 0.1  # seconds

# Distance between the robot position and the point seen at the center of
# the camera (arbitrarily set)
DELTA_D = 2.5
# Threshold for matching distance
EPSILON_D = 0.1
# Threshold for direction (robot should follow a straight line)
DELTA_ALPHA = 10.0 / 180.0 * np.pi

# Time window for gathering IMU data to estimate traversability
DELTA_T_IMU = 0.2


# Check for the script's arguments : python create_dataset.py ([filename].bag)

# Get the number of arguments passed to the script
nb_arguments = len(sys.argv)

# Too much arguments
if nb_arguments > 2:
    # Display an error message
    print("Invalid number of arguments: " + str(nb_arguments) + "\n")
    print("Should be 2: 'create_dataset.py' and '[bagfile_name].bag'")
    print("or just 1: 'create_dataset.py'")
    sys.exit(1)  # Error exit status

# Correct number of arguments
elif nb_arguments == 2:
    bag_files = [sys.argv[1]]
    nb_bags = len(bag_files)
    
    # Display a status message
    print(f"Reading only 1 bag file: {str(bag_files[0])}")

# Correct number of arguments
elif nb_arguments == 1:
    # Get the list of all the bag files located in the current directory
    bag_files = [f for f in os.listdir(".") if f[-4:] == ".bag"]
    nb_bags = len(bag_files)
    
    # Display a status message
    print(f"Reading all {nb_bags} bag file(s) located in current directory: \n")
    
    for bag_file in bag_files:
        print(bag_file)



# Dataset creation (data extracted from bag files)
file_count = 0  # File number in the list of bag files

# If there are several bag files, data are extracted from each of them
# and concatenated
for bag_file in bag_files:
    
    # Print file name and number
    file_count += 1
    print("Reading file " + str(file_count) +
          " of " + str(nb_bags) + ": " + bag_file)
    
    # Open the bag file
    bag = rosbag.Bag(bag_file)
    
    # Read the entire bag file
    # bag_contents = bag.read_messages()
    # Get the name of the file (without parent directories)
    bag_name = os.path.basename(bag.filename)
    
    
    # Create a new directory to store the dataset
    # Get the absolute path of the current directory
    directory = os.path.abspath(os.getcwd())
    
    # Set the name of the directory which will store the dataset
    results_dir = directory + "/datasets/dataset_" + bag_name[:-4]
    
    try:  # A new directory is created if it does not exist yet
        os.mkdir(results_dir)
        print(results_dir + " folder created\n")
        
    except OSError:  # Display a message if it already exists and quit
        print("Existing directory " + results_dir)
        print("Aborting\n")
        # sys.exit(1)  # Stop the execution of the script
        pass


    # Write images on disk and create a list of their timestamps
    # (when the image was published on the topic)
    full_image_list = []  # Will contain (timestamp of image, image name)

    bridge = CvBridge()  # To convert ROS Image type into numpy ndarray
    
    # Create a sub-directory to store images and their associated timestamp
    topic_name = IMAGE_TOPIC[1:].replace('/', '_')
    topic_dir = results_dir + "/" + topic_name

    # Create a directory if it does not exist yet
    try:
        os.mkdir(topic_dir)
        print(topic_dir + " folder created\n")
    except OSError:
        pass

    print("Writing images")

    # Create a csv file to store the timestamps
    csv_name = topic_dir + "/" + topic_name + ".csv"
    
    # Open the csv file and write the timestamps
    with open(csv_name, 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=',')

        # Write the first line of the file
        filewriter.writerow(["TimeStamp"])

        # Create a progress bar (its length is given by the number of images)
        pbar = tqdm(total=bag.get_message_count(IMAGE_TOPIC))
        
        # Go through the image topic
        for msg_index, (topic, msg, t) in enumerate(
            bag.read_messages(topics=IMAGE_TOPIC)):
            
            pbar.update()
            
            if msg_index == 0:
                t0 = t  # Save timestamp of the first image for later
            
            # We do not take all the images
            if msg_index % DOWNSAMPLING_RATIO == 0:
                
                # Convert the ROS Image to a numpy ndarray
                cv_img = bridge.imgmsg_to_cv2(
                    msg, desired_encoding="rgb8")
                
                # Save the image in the directory
                image = Image.fromarray(cv_img)
                image = image.convert('RGB')
                image_name = f"{msg_index:05d}.png"
                image.save(topic_dir + "/" +
                           image_name, "PNG")

                # Append the image name and the timestamp to the list
                full_image_list.append((rospy.Time.to_sec(t), image_name))
                
                # Write the timestamp in the csv file
                filewriter.writerow([str(t)])
                
        pbar.close()
        
        print(f"Wrote {int(bag.get_message_count(IMAGE_TOPIC)/DOWNSAMPLING_RATIO + 1)} images\n")



    # For each image, find the timestamp where the robot reaches the position
    # at the center of the image according to odometry,
    # get the corresponding IMU pitch measurement
    # and remove the image if nothing matches
    print("Filtering images that have corresponding positions")
    
    # Will contain (timestamp of image, image name, timestamp of position in front of robot)
    filtered_image_list = []
    last_t = t0  # Start from the timestamp of first image

    pbar = tqdm(total=len(full_image_list))
    
    # Go through the list of images
    for msg_index, (t_image, image_name) in enumerate(full_image_list):
        
        pbar.update()
        
        # Go through the odometry topic measurements
        for subtopic, msg, t in bag.read_messages(ODOM_TOPIC, start_time=last_t):
            if rospy.Time.to_sec(t) < t_image + EPSILON_T and rospy.Time.to_sec(t) > t_image - EPSILON_T:
                last_t = t

                # Get the position and the orientation (yaw only) of the robot
                # at the time of the image
                ref_position = msg.pose.pose.position
                orientation_q = msg.pose.pose.orientation
                quat_list = [orientation_q.x, orientation_q.y,
                             orientation_q.z, orientation_q.w]
                _, _, ref_yaw = euler_from_quaternion(quat_list)

                # Get the first future pose of the robot which matches the
                # fixed distance step
                for subtopic_2, msg_2, t_2 in bag.read_messages(ODOM_TOPIC,
                                                                start_time=t):
                    new_position = msg_2.pose.pose.position

                    # Motion direction since the reference pose
                    alpha = np.arctan2(
                        new_position.y - ref_position.y,
                        new_position.x - ref_position.x
                        ) - ref_yaw

                    # Travelled distance
                    dist = np.sqrt((new_position.x - ref_position.x)**2
                                   + (new_position.y - ref_position.y)**2)

                    # Keep only images for which we could extract the time
                    # the robot needed to travel the distance DELTA_D, and
                    # for which the robot followed an approximately straight
                    # trajectory
                    if abs(dist - DELTA_D) <= EPSILON_D and abs(alpha) < DELTA_ALPHA:
                        filtered_image_list.append(
                            (t_image, image_name, rospy.Time.to_sec(t_2)))
                        break
                    
                    elif dist > DELTA_D + EPSILON_D:
                        break
                break
            
    pbar.close()
    
    print(f"Remaining {len(filtered_image_list)} images that have required data for computing traversability\n")


    # For each remaining image, compute the variance of the pitch velocity
    # around the pose in front of the robot (at distance DELTA_D)
    print("Computing traversability")
    
    # Create a new csv file to store IMU data
    filename = results_dir + '/' + \
        str.replace(IMU_TOPIC[1:], '/', '_') + '.csv'

    # Write IMU data in this file
    with open(filename, 'w+') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=',')
        
        # Write the first row (columns title)
        headers = ["image_id", "y"]
        filewriter.writerow(headers)

        last_t = t0
        nb_valid_images = 0
        
        pbar = tqdm(total=len(filtered_image_list))
        
        # Go through the list of images, associated with the timestamp
        # for which the robot has traveled DELTA_D meters from its
        # current position
        for _, image_name, t_odom in filtered_image_list:
            
            pbar.update()
            
            # For each image, we get IMU data
            pitch_vel_list = []
            
            # Go through the IMU topic measurements
            for subtopic, msg, t in bag.read_messages(IMU_TOPIC,
                                                      start_time=last_t):
                
                # if rospy.Time.to_sec(t) < t_odom - DELTA_T_IMU:
                #     last_t = t
                
                # Keep IMU pitch value around the timestamp found with
                # odometry
                if abs(t_odom - rospy.Time.to_sec(t)) <= DELTA_T_IMU:
                    pitch_vel_list.append(msg.angular_velocity.y)
                    
                elif rospy.Time.to_sec(t) > t_odom + DELTA_T_IMU:
                    break

            if len(pitch_vel_list) > 2:
                # Compute the variance of the pitch velocity measures
                values = [image_name, np.var(pitch_vel_list)]
                
                # Write the image name and the pitch velocity variance in the
                # csv file
                filewriter.writerow(values)
                
                nb_valid_images += 1
                
            else:
                print("No values!")
    
    pbar.close()
    
    print(f"Remaining {nb_valid_images} images that have required data for traversability\n")

    bag.close()

print(f"Done reading all {nb_bags} bag files.")

end = time.time()

print(f"Processing time: {round(end - start, 1)} s.")
