"""
Script to build a dataset for terrain traversability estimation from images
(rgb, depth, normals) and the robot linear velocity. A dataset is a folder
with the following structure:

dataset_{name}/
├── images/
│   ├── 00000.npy
│   ├── 00000.tiff
│   ├── 00001.npy
│   ├── 00001.tiff
│   └── ...
├── images_test/
├── images_train/
├── traversal_costs.csv
├── traversal_costs_test.csv
├── traversal_costs_train.csv
└── bins_midpoints.csv

where:
- xxxxx.png and xxxxx.tiff are the rgb and depth images respectively
- images_train/ and images_test/ are the training and testing sets of images
- traversal_costs.csv is a csv file containing the traversal costs associated
with the images, the traversability labels (obtained from the continuous
traversal cost after digitization) and the linear velocities of the robot
- traversal_costs_train.csv and traversal_costs_test.csv contain the same
information but for the training and testing sets respectively
"""


# Python libraries
import numpy as np
import os
import csv
import sys
from tqdm import tqdm
import cv2
from PIL import Image
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True  # Render Matplotlib text with Tex
import pywt

# ROS Python libraries
import cv_bridge
import rosbag
import rospy
import tf.transformations

# Custom modules and packages
import utils.drawing as dw
import utils.frames as frames
import traversalcost.fourier.frequency_features as ff
import params.robot, params.dataset
import depth.utils as depth


def is_inside_image(image, point):
    """Check if a point is inside an image"""
    x, y = point
    return (x >= 0) and (x < image.shape[1]) and (y >= 0) and (y < image.shape[0])

def compute_features(signal, rate):
    
    # Moving average
    signal_filtered = uniform_filter1d(signal, size=3)
    
    # Variance
    signal_variance = np.var(signal_filtered)
    
    # Apply windowing because the signal is not periodic
    hanning_window = np.hanning(len(signal))
    signal_mean_windowing = signal_filtered*hanning_window
    
    # Discrete Fourier transform
    signal_magnitudes = np.abs(rfft(signal_mean_windowing))
    frequencies = rfftfreq(len(signal), rate)
    
    # Energies
    signal_energy = ff.spectral_energy(signal_magnitudes)
    
    # Spectral centroids
    signal_sc = ff.spectral_centroid(signal_magnitudes, frequencies)
    
    # Spectral spread
    signal_ss = ff.spectral_spread(signal_magnitudes, frequencies, signal_sc)
    
    return np.array([signal_variance, signal_energy, signal_sc, signal_ss])

def is_bag_healthy(bag):
    
    # Get the bag file duration
    duration = bag.get_end_time() - bag.get_start_time()  # [seconds]
    
    for topic, frequency in [(params.robot.IMAGE_TOPIC,
                              params.robot.CAMERA_SAMPLE_RATE),
                             (params.robot.DEPTH_TOPIC,
                              params.robot.DEPTH_SAMPLE_RATE),
                             (params.robot.ODOM_TOPIC,
                              params.robot.ODOM_SAMPLE_RATE),
                             (params.robot.IMU_TOPIC,
                              params.robot.IMU_SAMPLE_RATE)]:
        
        # Get the number of messages in the bag file
        nb_message = bag.get_message_count(topic)
        
        # Check if the number of messages is consistent with the frequency
        if np.abs(nb_message - frequency*duration)/(frequency*duration) > 0.01:
            return False
        
    return True


class DatasetBuilder(): 
    
    # Initialize the bridge between ROS and OpenCV images
    bridge = cv_bridge.CvBridge()

    # Create an array to store the features from which the traversal cost
    # is designed
    features = np.zeros((10000, params.dataset.NB_FEATURES))
    
    # Create an array to store the velocities
    velocities = np.zeros((10000, 1))
    
    # Create an array to store the pitch rate variance
    pitch_rate_variance = np.zeros((10000, 1))
    
    
    def __init__(self, name):
        
        # Get the absolute path of the current directory
        directory = os.path.abspath(os.getcwd())

        # Set the name of the directory which will store the dataset
        self.dataset_directory = directory + "/datasets/dataset_" + name

        try:  # A new directory is created if it does not exist yet
            os.mkdir(self.dataset_directory)
            print(self.dataset_directory + " folder created\n")

        except OSError:  # Display a message if it already exists and quit
            print("Existing directory " + self.dataset_directory)
            print("Aborting to avoid overwriting data\n")
            sys.exit(1)  # Stop the execution of the script
            # pass
        
        # Create a sub-directory to store images
        self.images_directory = self.dataset_directory + "/images"

        # Create a directory if it does not exist yet
        try:
            os.mkdir(self.images_directory)
            print(self.images_directory + " folder created\n")
        except OSError:
            pass
        
        # Create a csv file to store the traversal costs
        self.csv_name = self.dataset_directory + "/traversal_costs.csv"
        
    
    def write_images_and_compute_features(self, files):
        
        # Initialize the index of the image and the trajectory
        index_image = 0
        index_trajectory = 0
        
        # Go through multiple bag files
        for file in files:
            
            print("Reading file: " + file)
            
            # Open the bag file
            bag = rosbag.Bag(file)
            
            if not is_bag_healthy(bag):
                print("File " + file + " is incomplete. Skipping...")
                continue
            
            
            # Go through the image topic
            for _, msg_image, t_image in tqdm(
                bag.read_messages(topics=[params.robot.IMAGE_TOPIC]),
                total=bag.get_message_count(params.robot.IMAGE_TOPIC)):
                
                msg_depth = None
                TIME_DELTA = 0.05
                
                # Keep only the images that can be matched with a depth image
                if list(bag.read_messages(
                    topics=[params.robot.DEPTH_TOPIC],
                    start_time=t_image - rospy.Duration(TIME_DELTA),
                    end_time=t_image + rospy.Duration(TIME_DELTA))):
                    
                    # Find the depth image whose timestamp is closest to that
                    # of the rgb image
                    min_t = TIME_DELTA
                    
                    for _, msg_depth_i, t_depth in bag.read_messages(
                        topics=[params.robot.DEPTH_TOPIC],
                        start_time=t_image - rospy.Duration(TIME_DELTA),
                        end_time=t_image + rospy.Duration(TIME_DELTA)):
                        
                        if np.abs(t_depth.to_sec()-t_image.to_sec()) < min_t:
                            min_t = np.abs(t_depth.to_sec() - t_image.to_sec())
                            msg_depth = msg_depth_i
                
                else:
                    continue
                    
                # Get the first odometry message received after the image
                _, first_msg_odom, t_odom = next(iter(bag.read_messages(
                    topics=[params.robot.ODOM_TOPIC],
                    start_time=t_image)))
                
                # Collect images and IMU data only when the robot is moving
                # (to avoid collecting images of the same place) 
                if first_msg_odom.twist.twist.linear.x < \
                   params.dataset.LINEAR_VELOCITY_THR:
                    continue

                # Convert the current ROS image to the OpenCV type
                image = self.bridge.imgmsg_to_cv2(msg_image,
                                                  desired_encoding="passthrough")
                
                # Convert the current ROS depth image to the OpenCV type
                depth_image = self.bridge.imgmsg_to_cv2(msg_depth,
                                                  desired_encoding="passthrough")
                
                # Compute the transform matrix between the world and the robot
                WORLD_TO_ROBOT = frames.pose_to_transform_matrix(
                    first_msg_odom.pose.pose)
                # Compute the inverse transform
                ROBOT_TO_WORLD = frames.inverse_transform_matrix(WORLD_TO_ROBOT)

                # Initialize an array to store the previous front wheels
                # coordinates in the image
                points_image_old = None

                # Define a variable to store the previous timestamp
                t_odom_old = None
                
                point_world_old = None
                distance = 0
                
                x_velocity = []
                
                nb_rectangles = 0

                # Read the odometry measurement for T second(s)
                for i, (_, msg_odom, t_odom) in enumerate(bag.read_messages(
                    topics=[params.robot.ODOM_TOPIC],
                    start_time=t_odom,
                    end_time=t_odom+rospy.Duration(params.dataset.T))):

                    # Keep only an odometry measurement every dt second(s)
                    # if i % self.div != 0:
                    #     continue

                    # Store the 2D coordinates of the robot position in
                    # the world frame
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

                    # Create arrays to store left and write front
                    # wheels positions
                    point_left_world = np.copy(point_world)
                    point_right_world = np.copy(point_world)

                    # Compute the distances between the wheels and
                    # the robot's origin
                    delta_X = params.robot.L*np.sin(theta)/2
                    delta_Y = params.robot.L*np.cos(theta)/2

                    # Compute the positions of the outer points of the two
                    # front wheels
                    point_left_world[:, 0] -= delta_X
                    point_left_world[:, 1] += delta_Y
                    point_right_world[:, 0] += delta_X
                    point_right_world[:, 1] -= delta_Y

                    # Gather front wheels outer points coordinates in
                    # a single array
                    points_world = np.concatenate([point_left_world,
                                                   point_right_world])

                    # Compute the points coordinates in the robot frame
                    points_robot = frames.apply_rigid_motion(points_world,
                                                      ROBOT_TO_WORLD)

                    # Compute the points coordinates in the camera frame
                    points_camera = frames.apply_rigid_motion(
                        points_robot,
                        params.robot.CAM_TO_ROBOT)

                    # Compute the points coordinates in the image plan
                    points_image = frames.camera_frame_to_image(points_camera,
                                                                params.robot.K)
                    
                    # Test if the two points are inside the image
                    if not is_inside_image(image, points_image[0]) or \
                       not is_inside_image(image, points_image[1]):
                            if point_world_old is None:
                                continue
                            else:
                                # If the trajectory goes out of the image
                                break
                    
                    # First point in the image
                    if point_world_old is None:
                        # Draw the points on the image
                        # image = dw.draw_points(image, points_image)

                        point_world_old = point_world
                        points_image_old = points_image
                        t_odom_old = t_odom
                        continue
                        
                    distance = np.linalg.norm(point_world -
                                              point_world_old)
                    
                    # Get the linear velocity on x axis
                    x_velocity.append(msg_odom.twist.twist.linear.x)
                    
                    
                    if distance > params.dataset.PATCH_DISTANCE and \
                       nb_rectangles < params.dataset.NB_RECTANGLES_MAX:
                    
                        # Draw the points on the image
                        # image = dw.draw_points(image, points_image)
                        
                        # Compute the inclination of the patch in the image
                        delta_old = np.abs(points_image_old[0] - points_image_old[1])
                        delta_current = np.abs(points_image[0] - points_image[1])
                        
                        patch_angle = (
                            np.arctan(delta_old[1]/delta_old[0]) +
                            np.arctan(delta_current[1]/delta_current[0])
                            )/2
                        
                        # Discard rectangles that are too inclined
                        if patch_angle > params.dataset.PATCH_ANGLE_THR:
                            break
                        
                        # Compute the maximum and minimum coordinates on the
                        # y axis of the image plan
                        max_y = np.int32(np.max(points_image_old, axis=0)[1])
                        min_y = np.int32(np.min(points_image, axis=0)[1])

                        # Compute max and min coordinates of the points in
                        # the image along the x axis
                        min_x = np.int32(np.min([points_image_old[:, 0],
                                                 points_image[:, 0]]))
                        max_x = np.int32(np.max([points_image_old[:, 0],
                                                 points_image[:, 0]]))
                         
                        # Correct the dimensions of the rectangle to respect
                        # the height-width ratio
                        rectangle_width = max_x - min_x
                        rectangle_height = max_y - min_y
                        
                        if rectangle_width < \
                           params.dataset.RECTANGLE_RATIO*(rectangle_height):
                            # Height of the rectangular regions to be
                            # eliminated on the right and left of the
                            # patch
                            delta = np.int32((
                                rectangle_height -
                                (rectangle_width)/params.dataset.RECTANGLE_RATIO
                                )/2)
                            
                            # Coordinates of the vertices of the patch to keep
                            min_y_rectangle = min_y + delta
                            max_y_rectangle = max_y - delta
                            min_x_rectangle = min_x
                            max_x_rectangle = max_x
                        
                        else:
                            # Width of the rectangular regions to be
                            # eliminated at the top and bottom of the
                            # patch
                            delta = np.int32((
                                rectangle_width -
                                params.dataset.RECTANGLE_RATIO*(rectangle_height)
                                )/2)
                            
                            # Coordinates of the vertices of the patch to keep
                            min_x_rectangle = min_x + delta
                            max_x_rectangle = max_x - delta
                            min_y_rectangle = min_y
                            max_y_rectangle = max_y
                            
                        # Draw a rectangle in the image to visualize the region
                        # of interest
                        # image = dw.draw_quadrilateral(
                        #     image,
                        #     np.array([[min_x_rectangle, max_y_rectangle],
                        #               [min_x_rectangle, min_y_rectangle],
                        #               [max_x_rectangle, min_y_rectangle],
                        #               [max_x_rectangle, max_y_rectangle]]),
                        #     color=(255, 0, 0))
                        
                        # Set of points forming the quadrilateral
                        quadrilateral_points = np.array([
                            [points_image_old[0]],
                            [points_image_old[1]],
                            [points_image[1]],
                            [points_image[0]]
                        ], dtype=np.int32)
                        
                        # Compute the minimum area rectangle containing the set
                        rect = cv2.minAreaRect(quadrilateral_points)
                        
                        # Convert the rectangle to a set of 4 points
                        box = cv2.boxPoints(rect)
                        
                        # Draw the rectangle on the image
                        # image = dw.draw_quadrilateral(image,
                        #                               box,
                        #                               color=(0, 255, 0),
                        #                               thickness=1)
                        
                        # Extract the rectangular region of interest from
                        # the original image
                        image_to_save = image[min_y_rectangle:max_y_rectangle,
                                              min_x_rectangle:max_x_rectangle]
                        
                        # print("Image shape: ", image_to_save.shape)
                        # print("Image ratio: ", image_to_save.shape[1]/image_to_save.shape[0])
                        # print("Image points: ", points_image)
                        
                        # cv2.imshow('image', image_to_save)
                        # cv2.waitKey(0)
                        
                        # Convert the image from BGR to RGB
                        image_to_save = cv2.cvtColor(image_to_save,
                                                     cv2.COLOR_BGR2RGB)

                        # Make a PIL image
                        image_to_save = Image.fromarray(image_to_save)

                        # Give the image a name
                        image_name = f"{index_image:05d}.png"

                        # Save the image in the correct directory
                        image_to_save.save(self.images_directory + "/" + image_name, "PNG")
                        
                        # Extract the rectangular region of interest from
                        # the original depth image
                        depth_image_to_save = depth_image[min_y_rectangle:max_y_rectangle,
                                                          min_x_rectangle:max_x_rectangle].copy()
                        
                        # depth_image_example = cv2.imread("/home/tom/Traversability-Tom/Internship-U2IS/WuManchu_0360.png", cv2.IMREAD_ANYDEPTH)
                        
                        # normals_example = depth.compute_normals(depth_image_example)
                        
                        # normals_example = depth.convert_range(normals_example, 0, 1, 0, 255).astype(np.uint8)
                        # cv2.imshow('normals', normals_example)
                        # cv2.waitKey(0)
                        
                        # normals = depth.compute_normals(depth_image_to_save)
                        
                        # normals = depth.fill_nan_inf(normals)
                        
                        # normals = depth.convert_range(normals, 0, np.max(normals), 0, 255).astype(np.uint8)
                        # cv2.imshow('normals', normals)
                        # cv2.waitKey(0)
                        
                        # Replace NaN and Inf values (missing information) by a default value
                        depth_image_to_save = depth.fill_nan_inf(depth_image_to_save)
                        
                        # print(depth_image_to_save)
                        
                        # cv2.imshow("depth", depth.convert_range(depth_image_to_save,
                        #                                         0.7, 2, 0, 255).astype(np.uint8))
                        # cv2.waitKey(0)
                        
                        # Make a PIL image
                        depth_image_to_save = Image.fromarray(depth_image_to_save)
                        
                        # Give the depth image a name
                        depth_image_name = f"{index_image:05d}.tiff"
                        
                        # Save the depth image in the correct directory
                        depth_image_to_save.save(self.images_directory + "/" + depth_image_name, "TIFF")

                        nb_rectangles += 1
                        
                        # Create lists to store all the roll, pitch rate
                        # measurements within the dt second(s) interval
                        roll_velocity_values = []
                        pitch_velocity_values = []

                        # Create a list to store vertical acceleration values
                        z_acceleration_values = []
                        
                        # Read the IMU measurements within the dt second(s)
                        # interval
                        for _, msg_imu, _ in bag.read_messages(
                            topics=[params.robot.IMU_TOPIC],
                            start_time=t_odom_old,
                            end_time=t_odom):

                            # Append angular velocities and vertical
                            # acceleration to the previously created lists
                            roll_velocity_values.append(
                                msg_imu.angular_velocity.x)
                            pitch_velocity_values.append(
                                msg_imu.angular_velocity.y)
                            z_acceleration_values.append(
                                msg_imu.linear_acceleration.z - 9.81)

                        # print("roll: ", len(roll_velocity_values))
                        # print("pitch: ", np.var(pitch_velocity_values))
                        # print("z acceleration: ", len(z_acceleration_values))
                        # print("x velocity: ", np.mean(x_velocity))
                        
                        # Compute the variance of the pitch rate signal
                        self.pitch_rate_variance[index_image] = np.var(pitch_velocity_values)

                        # Pad the signals if they are too short
                        if len(roll_velocity_values) < params.dataset.SIGNAL_MIN_LENGTH:
                            pad = np.ceil((params.dataset.SIGNAL_MIN_LENGTH - len(roll_velocity_values))/2)

                            roll_velocity_values = pywt.pad(
                                roll_velocity_values,
                                pad_widths=pad,
                                mode=params.dataset.padding_mode)
                            pitch_velocity_values = pywt.pad(
                                pitch_velocity_values,
                                pad_widths=pad,
                                mode=params.dataset.padding_mode)
                            z_acceleration_values = pywt.pad(
                                z_acceleration_values,
                                pad_widths=pad,
                                mode=params.dataset.padding_mode)
                            
                        # Apply discrete wavelet transform
                        # approximation = coefficients[0]
                        # detail = coefficients[1:]
                        roll_coefficients  = pywt.wavedec(
                            roll_velocity_values,
                            level=params.dataset.NB_LEVELS,
                            wavelet=params.dataset.WAVELET)

                        pitch_coefficients  = pywt.wavedec(
                            pitch_velocity_values,
                            level=params.dataset.NB_LEVELS,
                            wavelet=params.dataset.WAVELET)

                        z_acceleration_coefficients  = pywt.wavedec(
                            z_acceleration_values,
                            level=params.dataset.NB_LEVELS,
                            wavelet=params.dataset.WAVELET)

                        # De-noising
                        for j in range(1, len(roll_coefficients)):
                            roll_coefficients[j] = pywt.threshold(
                                roll_coefficients[j],
                                value=params.dataset.DENOISE_THR,
                                mode="soft")
                            pitch_coefficients[j] = pywt.threshold(
                                pitch_coefficients[j],
                                value=params.dataset.DENOISE_THR,
                                mode="soft")
                            z_acceleration_coefficients[j] = pywt.threshold(
                                z_acceleration_coefficients[j],
                                value=params.dataset.DENOISE_THR,
                                mode="soft")
                            
                        # Fill the features array
                        for j in range(len(roll_coefficients)):
                            self.features[index_image, j] = np.var(
                                roll_coefficients[j])

                        for j in range(len(pitch_coefficients)):
                            self.features[index_image,
                                     len(roll_coefficients)+j] = np.var(
                                         pitch_coefficients[j])

                        for j in range(len(z_acceleration_coefficients)):
                            self.features[index_image,
                                     len(roll_coefficients)
                                     +len(pitch_coefficients)+j] = np.var(
                                         z_acceleration_coefficients[j])
                        
                        # Compute the mean velocity on the current patch
                        self.velocities[index_image] = np.mean(x_velocity)
                        
                        # self.features[index_image, 13] = index_trajectory
                        # self.features[index_image, 14] = t_image.to_sec() 

                        index_image += 1
                        
                        x_velocity = []
                        
                        point_world_old = point_world
                        points_image_old = points_image
                        t_odom_old = t_odom

                        # Display the image
                        # cv2.imshow("Image", image)
                        # cv2.waitKey()
                    
                    elif nb_rectangles == params.dataset.NB_RECTANGLES_MAX:
                        break
 

                index_trajectory += 1

            # Close the bag file
            bag.close()
        
        # Keep only the rows that are not filled with zeros
        self.features = self.features[np.any(self.features, axis=1)]
        self.velocities = self.velocities[np.any(self.velocities, axis=1)]
        self.pitch_rate_variance = self.pitch_rate_variance[np.any(self.pitch_rate_variance, axis=1)]
        
    def compute_traversal_costs(self):
         
        # # costs = self.features[:, 8, np.newaxis]
        # # self.features = self.features[:, [2, 6, 10]]
        # # self.features = self.features[:, [0, 4, 8]]
        # # features = self.features[:, [0, 2, 4, 6, 8, 10]]
        
        # # Scale the dataset
        # scaler = StandardScaler()
        # features_scaled = scaler.fit_transform(self.features)

        # # Apply PCA
        # pca = PCA(n_components=2)
        # costs = pca.fit_transform(features_scaled)
        
        # # Display the coefficients of the first principal component
        # plt.matshow(pca.components_, cmap="viridis")
        # plt.colorbar()
        # plt.xticks(range(12),
        #            [
        #                "var approx [roll]",
        #                "var lvl 1 [roll]",
        #                "var lvl 2 [roll]",
        #                "var lvl 3 [roll]",
        #                "var approx [pitch]",
        #                "var lvl 1 [pitch]",
        #                "var lvl 2 [pitch]",
        #                "var lvl 3 [pitch]",
        #                "var approx [z acc]",
        #                "var lvl 1 [z acc]",
        #                "var lvl 2 [z acc]",
        #                "var lvl 3 [z acc]",
        #             ],
        #            rotation=60,
        #            ha="left")
        # plt.xlabel("Feature")
        # plt.ylabel("Principal component 1")
        # plt.title("First principal component coefficients")
        
        # dataframe = pd.DataFrame(costs, columns=["pc1", "pc2"])
        # # dataframe["velocity"] = velocities

        # plt.figure()

        # plt.scatter(dataframe["pc1"],
        #             dataframe["pc2"],
        #             c=self.velocities[:, 0],
        #             cmap="jet")
        
        # plt.colorbar()

        # plt.xlabel("Principal component 1")
        # plt.ylabel("Principal component 2")
        
        # # robust_scaler = RobustScaler()
        # # normalized_costs = robust_scaler.fit_transform(costs)
        
        # # Polar
        # x = costs[:, 0]
        # x = x - np.min(x)
        # y = costs[:, 1]
        # r = np.sqrt(x**2 + y**2)
        # theta = np.arctan(y/(x+1e-3))
        # costs = r*np.sin((theta + np.pi/2)/2)
        
        # # Add an axis
        # costs = costs[:, None]
        
        # # Transform the costs to make the distribution closer
        # # to a Gaussian distribution
        # # normalized_costs = np.log(costs - np.min(costs) + 1)
        # normalized_costs = costs
        
        # # Create uniform bins
        # # nb_bins = 20
        # # bins = np.linspace(0, np.max(normalized_costs) + 1e-2, nb_bins+1)
        
        # # Distribution of the dataset
        # # plt.hist(normalized_costs, bins)
        
        # # Apply uniform binning (classes: from 0 to nb_bins - 1)
        # # digitized_costs = np.digitize(normalized_costs, bins) - 1
        
        # # # Apply one-hot encoding
        # # one_hot_encoder = OneHotEncoder()
        # # one_hot_costs = one_hot_encoder.fit_transform(digitized_costs).toarray()
        
        normalized_costs = self.pitch_rate_variance
        
        
        # Apply K-means binning
        discretizer = KBinsDiscretizer(
            n_bins=params.dataset.NB_BINS,
            encode="ordinal",
            strategy=params.dataset.BINNING_STRATEGY)
        digitized_costs = np.int32(discretizer.fit_transform(normalized_costs))
        
        # Get the edges and midpoints of the bins
        bins_edges = discretizer.bin_edges_[0]
        bins_midpoints = (bins_edges[:-1] + bins_edges[1:])/2
        
        # Save the midpoints in the dataset directory
        np.save(self.dataset_directory+"/bins_midpoints.npy", bins_midpoints)
        
        plt.figure()
        plt.hist(normalized_costs, bins_edges, lw=1, ec="magenta", fc="blue")
        plt.title("Traversal cost binning")
        plt.xlabel("Traversal cost")
        plt.ylabel("Samples")
        plt.show()
        
        return normalized_costs, digitized_costs

    def write_traversal_costs(self):
        
        # Open the file in the write mode
        file_costs = open(self.csv_name, "w")

        # Create a csv writer
        file_costs_writer = csv.writer(file_costs, delimiter=",")

        # Write the first row (columns title)
        headers = ["image_id", "traversal_cost", "traversability_label", "linear_velocity"]
        # headers = ["image_id", "traversal_cost", "traversability_label", "trajectory_id", "image_timestamp"]  #TODO:
        file_costs_writer.writerow(headers)
        
        costs, labels = self.compute_traversal_costs()
        # costs, labels, slip_costs = self.compute_traversal_costs()
        
        #TODO:
        # trajectories_ids = self.features[:, 13]
        # images_timestamps = self.features[:, 14]
        
        for i in range(costs.shape[0]):
            
            # Give the image a name
            image_name = f"{i:05d}"
            
            # cost = slip_costs[i, 0]
            cost = costs[i, 0]
            label = labels[i, 0]
            linear_velocity = self.velocities[i, 0]
            
            #TODO:
            # trajectory_id = trajectories_ids[i]
            # image_timestamp = images_timestamps[i]

            # Add the image index and the associated score in the csv file
            file_costs_writer.writerow([str(image_name), cost, label, linear_velocity])  #TODO:
            # file_costs_writer.writerow([image_name, cost, label, trajectory_id, image_timestamp])

        # Close the csv file
        file_costs.close()

    def create_train_test_splits(self):
        # Create a sub-directory to store train images
        train_directory = self.dataset_directory + "/images_train"
        os.mkdir(train_directory)
        print(train_directory + " folder created\n")
        
        # Create a sub-directory to store test images
        test_directory = self.dataset_directory + "/images_test"
        os.mkdir(test_directory)
        print(test_directory + " folder created\n")
        
        train_size = 0.85  # 85% of the data will be used for training
        
        # Read the CSV file into a Pandas dataframe (read image_id values as
        # strings to keep leading zeros)
        dataframe = pd.read_csv(self.csv_name, converters={"image_id": str})
        
        # Split the dataset randomly into training and testing sets
        dataframe_train, dataframe_test = train_test_split(dataframe, train_size=train_size)
        # dataframe_train, dataframe_test = train_test_split(dataframe,
        #                                                    train_size=train_size,
        #                                                    stratify=dataframe["traversability_label"])
        
        # Count the number of samples per class
        train_distribution = dataframe_train["traversability_label"].value_counts()
        test_distribution = dataframe_test["traversability_label"].value_counts()
        
        plt.bar(train_distribution.index, train_distribution.values, fc="blue", label="train")
        plt.bar(test_distribution.index, test_distribution.values, fc="orange", label="test")
        plt.legend()
        plt.title("Train and test sets distribution")
        plt.xlabel("Traversability label")
        plt.ylabel("Samples")
        plt.show()

        # Iterate over each row of the training set and copy the images to the training directory
        for _, row in dataframe_train.iterrows():
            image_file = os.path.join(self.images_directory, row["image_id"])
            shutil.copy(image_file + ".png", train_directory)
            shutil.copy(image_file + ".tiff", train_directory)

        # Iterate over each row of the testing set and copy the images to the testing directory
        for _, row in dataframe_test.iterrows():
            image_file = os.path.join(self.images_directory, row["image_id"])
            shutil.copy(image_file + ".png", test_directory)
            shutil.copy(image_file + ".tiff", test_directory)
        
        # Store the train and test splits in csv files
        dataframe_train.to_csv(self.dataset_directory + "/traversal_costs_train.csv", index=False)
        dataframe_test.to_csv(self.dataset_directory + "/traversal_costs_test.csv", index=False)
        

# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    dataset = DatasetBuilder(name="to_delete")
    
    dataset.write_images_and_compute_features(
        files=[
            # "bagfiles/raw_bagfiles/tom_1.bag",
            # "bagfiles/raw_bagfiles/tom_2.bag",
            # "bagfiles/raw_bagfiles/tom_3.bag",
            # "bagfiles/raw_bagfiles/tom_4.bag",
            # "bagfiles/raw_bagfiles/tom_5.bag",
            # "bagfiles/raw_bagfiles/tom_6.bag",
            # "bagfiles/raw_bagfiles/tom_7.bag",
            # "bagfiles/raw_bagfiles/tom_8.bag",
            # "bagfiles/raw_bagfiles/tom_path_grass.bag",
            # "bagfiles/raw_bagfiles/tom_grass_wood.bag",
            # "bagfiles/raw_bagfiles/tom_road.bag",
            # "bagfiles/raw_bagfiles/indoor.bag",
            # "bagfiles/raw_bagfiles/indoor2.bag",
            # "bagfiles/raw_bagfiles/simulation.bag"
            # "bagfiles/raw_bagfiles/depth/antoine_2.bag",
            "bagfiles/raw_bagfiles/depth/tom_missing.bag",
            "bagfiles/raw_bagfiles/depth/tom_full1.bag",
            "bagfiles/raw_bagfiles/depth/tom_full2.bag",
            "bagfiles/raw_bagfiles/depth/tom_full3.bag",
            "bagfiles/raw_bagfiles/depth/tom_full4.bag",
            "bagfiles/raw_bagfiles/depth/tom_full5.bag",
            "bagfiles/raw_bagfiles/depth/tom_full6.bag",
            "bagfiles/raw_bagfiles/depth/tom_full7.bag",
            "bagfiles/raw_bagfiles/depth/tom_full8.bag",
            "bagfiles/raw_bagfiles/depth/tom_full9.bag",
            "bagfiles/raw_bagfiles/depth/tom_full10.bag",
        ])

    dataset.write_traversal_costs()
    
    dataset.create_train_test_splits()
