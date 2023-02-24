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

# ROS Python libraries
import cv_bridge
import rosbag
import rospy
import tf.transformations

# Custom modules
import drawing as dw
import frames
import frequency_features as ff


class DatasetBuilder():
    
    # Topics name
    IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"
    ODOM_TOPIC = "/odometry/filtered"
    WHEELS_ODOM_TOPIC = "/husky_velocity_controller/odom"
    IMU_TOPIC = "imu/data"
    
    IMU_SAMPLE_RATE = 43
    
    # Time during which the future trajectory is taken into account
    T = 3  # seconds

    # Time step for trajectory sampling
    dt = 0.5  # seconds

    # Odometry measurements frequency
    ODOM_SAMPLE_RATE = 50  # Hz

    # Number of odometry measurements within dt second(s)
    div = np.int32(ODOM_SAMPLE_RATE*dt)

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
    # K = np.array([[700, 0, 640],
    #               [0, 700, 360],
    #               [0, 0, 1]])
    # K = np.array([[700, 0, 320],
    #               [0, 700, 180],
    #               [0, 0, 1]])

    # Initialize the bridge between ROS and OpenCV images
    bridge = cv_bridge.CvBridge()

    features = np.zeros((10000, 12))
    
    
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
        
        index_image = 0
        
        for file in files:
            
            print("Reading file: " + file)

            # Open the bag file
            bag = rosbag.Bag(file)
            

            # Go through the image topic
            for _, msg_image, t_image in tqdm(
                bag.read_messages(topics=[self.IMAGE_TOPIC]),
                total=bag.get_message_count(self.IMAGE_TOPIC)):

                # Convert the current ROS image to the OpenCV type
                image = self.bridge.imgmsg_to_cv2(msg_image,
                                                  desired_encoding="passthrough")

                # Get the first odometry message received after the image
                _, msg_odom, t_odom = next(iter(bag.read_messages(
                    topics=[self.ODOM_TOPIC],
                    start_time=t_image)))

                # Compute the transform matrix between the world and the robot
                WORLD_TO_ROBOT = frames.pose_to_transform_matrix(
                    msg_odom.pose.pose)
                # Compute the inverse transform
                ROBOT_TO_WORLD = frames.inverse_transform_matrix(WORLD_TO_ROBOT)

                # Initialize an array to store the previous front wheels
                # coordinates in the image
                points_image_old = 1e5*np.ones((2, 2))

                # Define a variable to store the previous timestamp
                t_odom_old = None

                # Read the odometry measurement for T second(s)
                for i, (_, msg_odom, t_odom) in enumerate(bag.read_messages(
                    topics=[self.ODOM_TOPIC],
                    start_time=t_odom,
                    end_time=t_odom+rospy.Duration(self.T))):

                    # Keep only an odometry measurement every dt second(s)
                    if i % self.div == 0:

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
                        delta_X = self.L*np.sin(theta)/2
                        delta_Y = self.L*np.cos(theta)/2

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
                            self.CAM_TO_ROBOT)

                        # Compute the points coordinates in the image plan
                        points_image = frames.camera_frame_to_image(points_camera,
                                                                    self.K)

                        # Draw the points on the image
                        # image = dw.draw_points(image, points_image)

                        # Compute the maximum and minimum coordinates on the
                        # y axis of the
                        # image plan
                        max_y = np.int32(np.max(points_image_old, axis=0)[1])
                        min_y = np.int32(np.min(points_image, axis=0)[1])


                        # Process the points only if they are visible in the image
                        if max_y < image.shape[0] and min_y > 0:

                            # Create lists to store all the roll, pitch angles and
                            # velocities measurement within the dt second(s)
                            # interval
                            # roll_angle_values = []
                            # pitch_angle_values = []
                            roll_velocity_values = []
                            pitch_velocity_values = []

                            z_acceleration_values = []

                            # Read the IMU measurements within the dt second(s)
                            # interval
                            for _, msg_imu, t_imu in bag.read_messages(
                                topics=[self.IMU_TOPIC],
                                start_time=t_odom_old,
                                end_time=t_odom):

                                # Append angles and angular velocities to the
                                # previously created lists
                                # roll_angle_values.append(
                                #     msg_imu.orientation.x)
                                # pitch_angle_values.append(
                                #     msg_imu.orientation.y)
                                roll_velocity_values.append(
                                    msg_imu.angular_velocity.x)
                                pitch_velocity_values.append(
                                    msg_imu.angular_velocity.y)
                                z_acceleration_values.append(
                                    msg_imu.linear_acceleration.z - 9.81)

                            # Compute max and min coordinates of the points in
                            # the image along the x axis
                            min_x = np.int32(np.min([points_image_old[:, 0],
                                                 points_image[:, 0]]))
                            max_x = np.int32(np.max([points_image_old[:, 0],
                                                 points_image[:, 0]]))

                            # Draw a rectangle in the image to visualize the region
                            # of interest
                            # image = dw.draw_quadrilateral(image,
                            #                            np.array([[min_x, max_y],
                            #                                      [min_x, min_y],
                            #                                      [max_x, min_y],
                            #                                      [max_x, max_y]]),
                            #                            color=(255, 0, 0))

                            # Extract the rectangular region of interest from
                            # the original image
                            image_to_save = image[min_y:max_y, min_x:max_x]


                            # Keep only rectangles which surface is greater than a
                            # given value
                            if image_to_save.shape[0]*image_to_save.shape[1] >= 10000 and \
                            image_to_save.shape[1]/image_to_save.shape[0] <= 5:

                                self.features[index_image] = self.compute_features(
                                    z_acceleration_values,
                                    roll_velocity_values,
                                    pitch_velocity_values)

                                # Convert the image from BGR to RGB
                                image_to_save = cv2.cvtColor(image_to_save,
                                                             cv2.COLOR_BGR2RGB)

                                # Make a PIL image
                                image_to_save = Image.fromarray(image_to_save)

                                # Give the image a name
                                image_name = f"{index_image:05d}.png"

                                # Save the image in the correct directory
                                image_to_save.save(self.images_directory + "/" + image_name, "PNG")

                                index_image += 1

                                # Display the image
                                # cv2.imshow("Image", image)
                                # cv2.waitKey()

                        # Update front wheels outer points image coordinates and timestamp
                        points_image_old = points_image
                        t_odom_old = t_odom


            # Close the bag file
            bag.close()
        
        # Keep only the rows that are not filled with zeros
        self.features = self.features[np.any(self.features, axis=1)]
        
    def compute_features(self,
                         z_acceleration_values,
                         roll_velocity_values,
                         pitch_velocity_values):
        
        # Moving average
        z_acceleration_filtered = uniform_filter1d(z_acceleration_values, size=3)
        roll_velocity_filtered = uniform_filter1d(roll_velocity_values, size=3)
        pitch_velocity_filtered = uniform_filter1d(pitch_velocity_values, size=3)

        # Variances
        z_acceleration_variance = np.var(z_acceleration_filtered)
        roll_velocity_variance = np.var(roll_velocity_filtered)
        pitch_velocity_variance = np.var(pitch_velocity_filtered)

        # Apply windowing because the signal is not periodic
        hanning_window = np.hanning(len(z_acceleration_values))
        z_acceleration_mean_windowing = z_acceleration_filtered*hanning_window
        roll_velocity_mean_windowing = roll_velocity_filtered*hanning_window
        pitch_velocity_mean_windowing = pitch_velocity_filtered*hanning_window

        # Discrete Fourier transform
        z_acceleration_magnitudes = np.abs(rfft(z_acceleration_mean_windowing))
        roll_velocity_magnitudes = np.abs(rfft(roll_velocity_mean_windowing))
        pitch_velocity_magnitudes = np.abs(rfft(pitch_velocity_mean_windowing))
        frequencies = rfftfreq(len(z_acceleration_values), 1/self.IMU_SAMPLE_RATE)

        # Energies
        z_acceleration_energy = ff.spectral_energy(z_acceleration_magnitudes)
        roll_velocity_energy = ff.spectral_energy(roll_velocity_magnitudes)
        pitch_velocity_energy = ff.spectral_energy(pitch_velocity_magnitudes)

        # Spectral centroids
        z_acceleration_sc = ff.spectral_centroid(z_acceleration_magnitudes, frequencies)
        roll_velocity_sc = ff.spectral_centroid(roll_velocity_magnitudes, frequencies)
        pitch_velocity_sc = ff.spectral_centroid(pitch_velocity_magnitudes, frequencies)

        # Spectral spread
        z_acceleration_ss = ff.spectral_spread(z_acceleration_magnitudes, frequencies, z_acceleration_sc)
        roll_velocity_ss = ff.spectral_spread(roll_velocity_magnitudes, frequencies, roll_velocity_sc)
        pitch_velocity_ss = ff.spectral_spread(pitch_velocity_magnitudes, frequencies, pitch_velocity_sc)

        
        return np.array([z_acceleration_variance, z_acceleration_energy, z_acceleration_sc, z_acceleration_ss,
                         roll_velocity_variance, roll_velocity_energy, roll_velocity_sc, roll_velocity_ss,
                         pitch_velocity_variance, pitch_velocity_energy, pitch_velocity_sc, pitch_velocity_ss])
    
    def compute_traversal_costs(self):
        
        # costs = self.features[:, 8, np.newaxis]
        # self.features = self.features[:, [2, 6, 10]]
        # self.features = self.features[:, [0, 4, 8]]
        self.features = self.features[:, [0, 2, 4, 6, 8, 10]]
        
        # Scale the dataset
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)

        # Apply PCA
        pca = PCA(n_components=1)
        costs = pca.fit_transform(features_scaled)
        
        # Display the coefficients of the first principal component
        # plt.matshow(pca.components_, cmap="viridis")
        # plt.colorbar()
        # plt.xticks(range(12),
        #            ["variance (z acceleration)", "energy (z acceleration)",
        #             "spectral centroid (z acceleration)",
        #             "spectral spread (z acceleration)",
        #             "variance (roll rate)", "energy (roll rate)",
        #             "spectral centroid (roll rate)",
        #             "spectral spread (roll rate)",
        #             "variance (pitch rate)", "energy (pitch rate)",
        #             "spectral centroid (pitch rate)",
        #             "spectral spread (pitch rate)"],
        #            rotation=60,
        #            ha="left")
        # plt.xlabel("Feature")
        # plt.ylabel("Principal component 1")
        # plt.title("First principal component coefficients")
        # plt.show()
        
        # robust_scaler = RobustScaler()
        # normalized_costs = robust_scaler.fit_transform(costs)
        
        # Transform the costs to make the distribution closer
        # to a Gaussian distribution
        normalized_costs = np.log(costs - np.min(costs) + 1)
        # normalized_costs = costs
        
        # Create uniform bins
        # nb_bins = 20
        # bins = np.linspace(0, np.max(normalized_costs) + 1e-2, nb_bins+1)
        
        # Distribution of the dataset
        # plt.hist(normalized_costs, bins)
        
        # Apply uniform binning (classes: from 0 to nb_bins - 1)
        # digitized_costs = np.digitize(normalized_costs, bins) - 1
        
        # # Apply one-hot encoding
        # one_hot_encoder = OneHotEncoder()
        # one_hot_costs = one_hot_encoder.fit_transform(digitized_costs).toarray()
        
        # Apply K-means binning
        nb_bins = 10  # Number of bins
        discretizer = KBinsDiscretizer(n_bins=nb_bins, encode="ordinal", strategy="kmeans")
        digitized_costs = np.int32(discretizer.fit_transform(normalized_costs))
        
        # Get the edges and midpoints of the bins
        bins_edges = discretizer.bin_edges_[0]
        bins_midpoints = (bins_edges[:-1] + bins_edges[1:])/2
        # print(f"Bins midpoints: {bin_midpoints}\n"
        
        # Save the midpoints in the dataset directory
        np.save(self.dataset_directory+"/bins_midpoints.npy", bins_midpoints)
        
        plt.hist(normalized_costs, bins_edges, lw=1, ec="magenta", fc="blue")
        plt.title("Traversal cost binning")
        plt.show()
        
        return normalized_costs, digitized_costs

    def write_traversal_costs(self):
        
        # Open the file in the write mode
        file_costs = open(self.csv_name, "w")

        # Create a csv writer
        file_costs_writer = csv.writer(file_costs, delimiter=",")

        # Write the first row (columns title)
        headers = ["image_id", "traversal_cost", "traversability_label"]
        file_costs_writer.writerow(headers)
        
        costs, labels = self.compute_traversal_costs()
        
        for i in range(costs.shape[0]):
            
            # Give the image a name
            image_name = f"{i:05d}.png"
            
            cost = costs[i, 0]
            label = labels[i, 0]

            # Add the image index and the associated score in the csv file
            file_costs_writer.writerow([image_name, cost, label])

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
        
        # Read the CSV file into a Pandas dataframe
        dataframe = pd.read_csv(self.csv_name)

        # Split the dataset randomly into training and testing sets
        dataframe_train, dataframe_test = train_test_split(dataframe, train_size=train_size, stratify=dataframe["traversability_label"])
        
        # Count the number of samples per class
        train_distribution = dataframe_train["traversability_label"].value_counts()
        test_distribution = dataframe_test["traversability_label"].value_counts()
        
        plt.bar(train_distribution.index, train_distribution.values, fc="blue", label="train")
        plt.bar(test_distribution.index, test_distribution.values, fc="orange", label="test")
        plt.legend()
        plt.title("Train and test sets distribution")
        plt.show()

        # Iterate over each row of the training set and copy the images to the training directory
        for _, row in dataframe_train.iterrows():
            image_file = os.path.join(self.images_directory, row["image_id"])
            shutil.copy(image_file, train_directory)

        # Iterate over each row of the testing set and copy the images to the testing directory
        for _, row in dataframe_test.iterrows():
            image_file = os.path.join(self.images_directory, row["image_id"])
            shutil.copy(image_file, test_directory)
        
        # Store the train and test splits in csv files
        dataframe_train.to_csv(self.dataset_directory + "/traversal_costs_train.csv", index=False)
        dataframe_test.to_csv(self.dataset_directory + "/traversal_costs_test.csv", index=False)
        

# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    dataset = DatasetBuilder(name="3+8bags_3var3sc_regression_classification_kmeans_split")
    
    dataset.write_images_and_compute_features(
        files=[
            "bagfiles/raw_bagfiles/tom_1.bag",
            "bagfiles/raw_bagfiles/tom_2.bag",
            "bagfiles/raw_bagfiles/tom_3.bag",
            "bagfiles/raw_bagfiles/tom_4.bag",
            "bagfiles/raw_bagfiles/tom_5.bag",
            "bagfiles/raw_bagfiles/tom_6.bag",
            "bagfiles/raw_bagfiles/tom_7.bag",
            "bagfiles/raw_bagfiles/tom_8.bag",
            "bagfiles/raw_bagfiles/tom_path_grass.bag",
            "bagfiles/raw_bagfiles/tom_grass_wood.bag",
            "bagfiles/raw_bagfiles/tom_road.bag",
        ])

    dataset.write_traversal_costs()
    
    dataset.create_train_test_splits()
