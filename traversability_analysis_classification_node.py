#!/usr/bin/env python3

# ROS - Python librairies
import rospy
import cv_bridge
import tf.transformations

# Import useful ROS types
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

# Python librairies
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import PIL
import time

# Custom modules
import frames
import drawing as dw


class TraversabilityAnalysis:
    
    # Topics name
    IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"
    ODOM_TOPIC = "/odometry/filtered"
    
    IMU_SAMPLE_RATE = 43
    
    # Image dimensions
    IMAGE_H, IMAGE_W = 720, 1280
    # IMAGE_H, IMAGE_W = 360, 640
    
    # Time for which the future trajectory predicted
    T = 3  # seconds

    # Integration step
    dt = 0.05  # seconds

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
    # K = np.array([[500, 0, 320],
    #               [0, 500, 180],
    #               [0, 0, 1]])
    
    # Distorsion coefficients
    # [k1, k2, p1, p2, k3]
    distorsion = np.array([-0.041, 0.010, 0.000, 0.000, -0.005])
    
    # Compute the new calibration matrix when distorsion is removed
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, distorsion, (IMAGE_W,IMAGE_H), 1, (IMAGE_W,IMAGE_H))
    x_cropped, y_cropped, w_cropped, h_cropped = roi
    print(new_K)
    print(roi)
    
    # Compute the transform matrix between the world and the robot
    WORLD_TO_ROBOT = np.eye(4)  # The trajectory is directly generated in the robot frame

    # Compute the inverse transform
    ROBOT_TO_WORLD = frames.inverse_transform_matrix(WORLD_TO_ROBOT)

    # Initialize the bridge between ROS and OpenCV images
    bridge = cv_bridge.CvBridge()
    
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device, "\n")
    
    # Load the pre-trained model
    model = models.resnet18().to(device=device)

    # Replace the last layer by a fully-connected one with 1 output
    model.fc = nn.Linear(model.fc.in_features, 10, device=device)
    
    # Load the fine-tuned weights
    model.load_state_dict(torch.load("src/models_development/models_parameters/resnet18_fine_tuned_3+8bags_3var3sc_classification_kmeans_100epochs.params"))

    model.eval()

    transform = transforms.Compose([ 
        # transforms.Resize(100),
        transforms.Resize((70, 210)),
        # transforms.Grayscale(),
        # transforms.CenterCrop(100),
        # transforms.RandomCrop(100),
        transforms.ToTensor(),
        # Mean and standard deviation were pre-computed on the training data
        # (on the ImageNet dataset)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Midpoints of the bins
    bin_midpoints = torch.tensor([0.43156512, 0.98983318, 1.19973744, 1.35943443, 1.51740755,
                                  1.67225206, 1.80821536, 1.94262708, 2.12798895, 2.6080252], device=device)
    # Add a dimension to the tensor
    bin_midpoints = bin_midpoints[:, None]

    
    def __init__(self):
        """Constructor of the class
        """
        # Initialize the subscriber to the image topic
        self.sub_image = rospy.Subscriber("/zed_node/rgb/image_rect_color", Image,
                                          self.callback_image, queue_size=1)
        
        self.sub_odom = rospy.Subscriber("odometry/filtered", Odometry,
                                         self.callback_odom, queue_size=1)

    def motion(self, x, u, dt):
        """
        Motion model (backward Euler method
        applied to differential drive kinematic model)
        """

        x[2] += u[1] * dt
        x[0] += u[0] * np.cos(x[2]) * dt
        x[1] += u[0] * np.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x

    def predict_trajectory(self, x_init, v, omega, predict_time=3.0, dt=0.05):
        """
        Predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= predict_time:
            x = self.motion(x, [v, omega], dt)
            trajectory = np.vstack((trajectory, x))
            time += dt

        return trajectory
    
    def compute_points_image(self, trajectory):
        # Get the robot successive poses (x, y, theta)
        points_world = np.zeros((trajectory.shape[0], 3))
        points_world[:, :2] = trajectory[:, :2]
        theta = trajectory[:, 2]
    
        # Create arrays to store left and write front wheels positions
        points_left_world = np.copy(points_world)
        points_right_world = np.copy(points_world)
    
        # Compute the distances between the wheels and the robot's origin
        delta_X = self.L*np.sin(theta)/2
        delta_Y = self.L*np.cos(theta)/2
    
        # Compute the positions of the outer points of the two
        # front wheels
        points_left_world[:, 0] -= delta_X
        points_left_world[:, 1] += delta_Y
        points_right_world[:, 0] += delta_X
        points_right_world[:, 1] -= delta_Y
        
        # Reverse points order
        points_right_world = points_right_world[::-1]
        
        # Gather front wheels outer points coordinates in a single array
        points_world = np.concatenate([points_left_world,
                                       points_right_world])
    
        # Compute the points coordinates in the camera frame
        points_camera = frames.apply_rigid_motion(
            points_world,
            np.dot(self.CAM_TO_ROBOT, self.ROBOT_TO_WORLD))
    
        # Compute the points coordinates in the image plan
        points_image = frames.camera_frame_to_image(points_camera,
                                             self.K)
         
        return points_image
    
    def remove_outside_points(self, points_image):
        """Remove the points which are outside the image

        Args:
            points_image (ndarray (n, 2)): Points coordinates in the image plan

        Returns:
            ndarray(n', 2): Points which are inside the image (n' <= n)
        """
        # Keep only points which are on the image
        is_inside = (points_image[:, 0] > 0) & (points_image[:, 0] < self.IMAGE_W) & (points_image[:, 1] > 0) & (points_image[:, 1] < self.IMAGE_H)
        
        points_image = points_image[is_inside]
        
        return points_image
    
    def remove_outside_pairs(self, points_image):
        """Remove left and right wheels points, where at least one of them is outside the image

        Args:
            points_image (ndarray (n, 2)): Points coordinates in the image plan

        Returns:
            ndarray (n', 2): Pairs of points which are inside the image (n - n' = 0 [2])
        """
        
        # Keep only points which are on the image
        is_inside = (points_image[:, 0] > 0) & (points_image[:, 0] < self.IMAGE_W) & (points_image[:, 1] > 0) & (points_image[:, 1] < self.IMAGE_H)
        
        are_both_inside = is_inside & is_inside[::-1]
        
        points_image = points_image[are_both_inside]
        
        return points_image
    
    def display_trajectory(self, image, trajectory_image, color=(255, 0, 0), transparency=0.3):
        
        # Create an overlay to segment the robot path
        overlay = np.copy(image)

        # Represent the path as a filled polygon
        trajectory_image = trajectory_image.astype(np.int32)
        trajectory_image = trajectory_image.reshape((-1, 1, 2))
        overlay = cv2.fillPoly(overlay, [trajectory_image], color)
        
        # Weighted sum of the original image and the overlay to create transparency
        image = cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0)
        
        return image
    
    def linear_gradient(self, cost, cost_min, cost_max):
        green = 255/(cost_min - cost_max)*cost + 255*cost_max/(cost_max - cost_min)
        red = 255 - green

        return (0, green, red)
    
    def predict_rectangle_cost(self, rectangle, model, transform):
        
        # Turn off gradients computation
        with torch.no_grad():
            
            # Convert the image from BGR to RGB
            rectangle = cv2.cvtColor(rectangle, cv2.COLOR_BGR2RGB)
            # Make a PIL image
            rectangle = PIL.Image.fromarray(rectangle)

            # Apply transforms to the image
            rectangle = transform(rectangle)

            # Add a dimension of size one (to create a batch of size one)
            rectangle = torch.unsqueeze(rectangle, dim=0)
            rectangle = rectangle.to(self.device)
            cost = torch.argmax(nn.Softmax()(model(rectangle))).item()
        
        return cost
    
    def predict_rectangles_costs(self, rectangles, model, transform):
        
        # Turn off gradients computation
        with torch.no_grad():
            
            # Define an empty list to create the batch of images
            rectangles_batch = []
            
            for rectangle in rectangles:
                time_pred_start = time.time()

                # Convert the image from BGR to RGB
                rectangle = cv2.cvtColor(rectangle, cv2.COLOR_BGR2RGB)
                # Make a PIL image
                rectangle = PIL.Image.fromarray(rectangle)

                # Apply transforms to the image
                rectangle = transform(rectangle )

                # Add a dimension of size one (to create a batch of size one)
                rectangle = torch.unsqueeze(rectangle, dim=0)
                
                rectangles_batch.append(rectangle)
                
                time_pred_stop = time.time()
            
            # Concatenate all the cropped rectangular images to make the batch
            rectangles_batch = torch.cat(rectangles_batch, 0)
            
            rectangles_batch = rectangles_batch.to(self.device)
            
            # Compute the expected traversal cost
            costs = torch.matmul(nn.Softmax()(model(rectangles_batch)), self.bin_midpoints)
            
            # Get the class with the highest probability
            # costs = torch.argmax(nn.Softmax()(model(rectangles_batch)), dim=1)
            
            print("Prediction time: ", time_pred_stop - time_pred_start)
        
        return costs
    
    def extract_trajectory_rectangles(self, trajectory_image, image):
        
        nb_points = trajectory_image.shape[0]
        assert nb_points % 2 == 0, "There must be an even number of points"
        
        # Compute the number of quadrilaterals that can be drawn
        nb_quadrilaterals = np.int32(nb_points/2 - 1)
        
        points_quadrilateral = np.zeros((4, 2))
        
        rectangles = []
        
        for i in range(nb_quadrilaterals):
            # Fill the coordinates of each quadrilaterals
            points_quadrilateral[:2] = trajectory_image[i:i+2]
            points_quadrilateral[-2:] = trajectory_image[2*nb_quadrilaterals-i:nb_points-i]
            
            # Compute the bounds of the bounding box
            min_x, min_y = np.int32(np.min(points_quadrilateral, axis=0))
            max_x, max_y = np.int32(np.max(points_quadrilateral, axis=0))
            # image = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color=(255, 0, 0))
            
            # Extract the rectangular region from the image
            rectangle = image[min_y:max_y, min_x:max_x]
            
            # Append the rectangle to the list of rectangles
            rectangles.append(rectangle)
            
        return rectangles, nb_quadrilaterals
     
    def predict_trajectory_cost(self, trajectory_rectangles):
        
        # Predict the traversal cost of the rectangular regions
        costs = self.predict_rectangles_costs(trajectory_rectangles, self.model, self.transform)
        
        # image = dw.draw_points(image, trajectory_image)
        
        # Compute the cost of the trajectory
        # cost = np.mean(costs)
        cost = torch.max(costs).item()
        
        return cost
    
    def predict_trajectories_costs(self, trajectories_rectangles, nb_rectangles_trajectories):
        
        # Predict the traversal cost of the rectangular regions
        costs = self.predict_rectangles_costs(trajectories_rectangles, self.model, self.transform)
        
        # Define an empty list to store the trajectories costs
        trajectories_costs = []
        
        indexes = np.cumsum(nb_rectangles_trajectories) - nb_rectangles_trajectories[0]
        
        # Compute the traversal cost for each trajectory
        for i in range(len(nb_rectangles_trajectories)):
            trajectories_costs.append(
                torch.max(  #TODO:
                    costs[indexes[i]:indexes[i]+nb_rectangles_trajectories[i]]
                ).item())
        
        return trajectories_costs
    
    
    def callback_image(self, msg):
        """Function called each time a new ros Image message is received on
        the image topic
        Args:
            msg (sensor_msgs/Image): a ROS image sent by the camera
        """
        
        time_start = time.time()
        
        # Convert the ROS Image into the OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image_to_display = np.copy(image)
        
        # image_undistorted_cropped = cv2.undistort(image_to_display, self.K, self.distorsion, None, self.new_K)[self.y_cropped:self.y_cropped+self.h_cropped, self.x_cropped:self.x_cropped+self.w_cropped]
        
        # Current state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        x_current = [0., 0., 0., self.v, self.omega]
        # x_current = [0., 0., self.theta, self.v, self.omega]
        
        # Set the list of angular velocities
        omegas = [0.5, 0.3, 0., -0.3, -0.5]
        # omegas = [0.5]
        
        # Define an empty list to store the trajectories cost
        # trajectories_costs = []
        
        # Define an empty list to store all the rectangular regions
        # cropped in the image
        trajectories_rectangles = []
        
        nb_rectangles_trajectories = []
        
        # Define an empty list to store the trajectories
        trajectories = []
        
        # Go through the linear / angular velocities candidate couples
        # (the linear velocity is always taken equal to 1 here)
        for omega in omegas:
            # Predict a trajectory given the current state of the robot
            # and velocities
            trajectory = self.predict_trajectory(x_current, v=1., omega=omega, predict_time=self.T, dt=self.dt)
            trajectories.append(trajectory)
            
            # Project the points expressed in the world frame onto the image plan
            # trajectory_image = self.compute_points_image(trajectory)
            trajectory_image_sampled = self.compute_points_image(trajectory[::8])
            
            # Remove points or pairs of points which are outside the image
            # trajectory_image = self.remove_outside_points(trajectory_image)
            trajectory_image_sampled = self.remove_outside_pairs(trajectory_image_sampled)
            
            # Extract the rectangles in the trajectory from the image
            trajectory_rectangles, nb_rectangles = self.extract_trajectory_rectangles(trajectory_image_sampled, image)
            # Concatenate the rectangles
            trajectories_rectangles += trajectory_rectangles
            #
            nb_rectangles_trajectories.append(nb_rectangles)
            
            # Compute the trajectory cost
            # trajectory_cost = self.predict_trajectory_cost(trajectory_rectangles)
            # trajectories_costs.append(trajectory_cost)
            
            # Get the color associated with a cost
            # color = self.linear_gradient(trajectory_cost, -2, 5)
            
            # Display the trajectory on the image
            # image_to_display = self.display_trajectory(image_to_display, trajectory_image, color=color)
        
        # Compute the costs of all the rectangular regions of the image and
        # gather them into trajectories costs
        trajectories_costs = self.predict_trajectories_costs(trajectories_rectangles, nb_rectangles_trajectories)
        
        # print("\n")
        
        # List of text positions on the x axis
        text_x = [210, 410, 610, 810, 1010]
        
        # Go through the trajectories
        for i in range(len(omegas)):
            
            # Project the points expressed in the world frame onto the image plan
            trajectory_image = self.compute_points_image(trajectories[i])
            
            # Remove points or pairs of points which are outside the image
            trajectory_image = self.remove_outside_points(trajectory_image)
            
            # Compute the color associated with the cost
            color_trajectory = self.linear_gradient(trajectories_costs[i], 0, 2.5)
            # color_trajectory = self.linear_gradient(trajectories_costs[i], 0, 9)
            
            # Display the trajectory on the image
            image_to_display = self.display_trajectory(image_to_display, trajectory_image, color=color_trajectory)
        
        for i in range(len(omegas)):
            # Set the color blue for the lowest trajectory cost
            color_text = (255, 0, 0) if trajectories_costs[i] == np.min(trajectories_costs) else (255, 255, 255)  # TODO:
            
            # Put the trajectory cost on the image
            image_to_display = cv2.putText(image_to_display,
                                           str(np.round(trajectories_costs[i], 2)),
                                           org=(text_x[i], 700),
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1,
                                           color=color_text,
                                           thickness=2)
            
        
        cv2.imshow("Image", image_to_display)
        cv2.waitKey(2)
        
        time_stop = time.time()
        print("Execution time: ", time_stop - time_start, "seconds")

    def callback_odom(self, msg):
        # Make the orientation quaternion a numpy array
        q = np.array([msg.pose.pose.orientation.x,
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z,
                      msg.pose.pose.orientation.w])
        
        # Convert the quaternion into Euler angles
        self.theta = tf.transformations.euler_from_quaternion(q)[2]
        
        # Get the linear velocity of the robot
        self.v = np.linalg.norm([msg.twist.twist.linear.x,
                                 msg.twist.twist.linear.y])
        
        # Get the angular velocity of the robot
        self.omega = msg.twist.twist.angular.z


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("traversability_analysis")

    # Instantiate an object
    traversability_analysis = TraversabilityAnalysis()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
