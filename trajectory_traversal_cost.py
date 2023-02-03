import numpy as np
import cv2
import rosbag
import tf.transformations
import cv_bridge
import rospy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


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

def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x

def predict_trajectory(x_init, v, y, predict_time=3.0, dt=0.1):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= predict_time:
        x = motion(x, [v, y], dt)
        trajectory = np.vstack((trajectory, x))
        time += dt

    return trajectory

def eval_trajectory(image, trajectory):
    points_world = np.zeros((trajectory.shape[0], 3))
    points_world[:, :2] = trajectory[:, :2]
    theta = trajectory[:, 2]

    # Create arrays to store left and write front wheels positions
    points_left_world = np.copy(points_world)
    points_right_world = np.copy(points_world)

    # Compute the distances between the wheels and the robot's origin
    delta_X = L*np.sin(theta)/2
    delta_Y = L*np.cos(theta)/2

    # Compute the positions of the outer points of the two
    # front wheels
    points_left_world[:, 0] -= delta_X
    points_left_world[:, 1] += delta_Y
    points_right_world[:, 0] += delta_X
    points_right_world[:, 1] -= delta_Y

    # Gather front wheels outer points coordinates in a single array
    points_world = np.concatenate([points_left_world,
                                   points_right_world])

    # Compute the points coordinates in the robot frame
    points_robot = apply_rigid_motion(points_world,
                                      ROBOT_TO_WORLD)

    # Compute the points coordinates in the camera frame
    points_camera = apply_rigid_motion(points_robot,
                                       CAM_TO_ROBOT)

    # Compute the points coordinates in the image plan
    points_image = camera_frame_to_image(points_camera,
                                         K)

    
    points_image = points_image[::5]
    
    
    points_quadri1 = np.vstack([points_image[5:7], points_image[-2:]])
    min_x1 = np.int32(np.min(points_quadri1[:, 0], axis=0))
    max_x1 = np.int32(np.max(points_quadri1[:, 0], axis=0))
    min_y1 = np.int32(np.min(points_quadri1[:, 1], axis=0))
    max_y1 = np.int32(np.max(points_quadri1[:, 1], axis=0))
    
    points_quadri2 = np.vstack([points_image[4:6], points_image[-3:-1]])
    min_x2 = np.int32(np.min(points_quadri2[:, 0], axis=0))
    max_x2 = np.int32(np.max(points_quadri2[:, 0], axis=0))
    min_y2 = np.int32(np.min(points_quadri2[:, 1], axis=0))
    max_y2 = np.int32(np.max(points_quadri2[:, 1], axis=0))
    
    points_quadri3 = np.vstack([points_image[3:5], points_image[-4:-2]])
    min_x3 = np.int32(np.min(points_quadri3[:, 0], axis=0))
    max_x3 = np.int32(np.max(points_quadri3[:, 0], axis=0))
    min_y3 = np.int32(np.min(points_quadri3[:, 1], axis=0))
    max_y3 = np.int32(np.max(points_quadri3[:, 1], axis=0))
    
    # image_to_display = draw_quadrilateral(image_to_display,
    #                                       np.array([[min_x1, max_y1],
    #                                                 [min_x1, min_y1],
    #                                                 [max_x1, min_y1],
    #                                                 [max_x1, max_y1]]),
    #                                       color=(255, 0, 0))
    # image_to_display = draw_quadrilateral(image_to_display,
    #                                       np.array([[min_x2, max_y2],
    #                                                 [min_x2, min_y2],
    #                                                 [max_x2, min_y2],
    #                                                 [max_x2, max_y2]]),
    #                                       color=(255, 0, 0))
    # image_to_display = draw_quadrilateral(image_to_display,
    #                                       np.array([[min_x3, max_y3],
    #                                                 [min_x3, min_y3],
    #                                                 [max_x3, min_y3],
    #                                                 [max_x3, max_y3]]),
    #                                       color=(255, 0, 0))
    
    # # Draw the points on the image
    # # image_to_display = draw_points(np.copy(image), points_image)
    # image_to_display = draw_points(image_to_display, points_image)

    # cv2.imshow("Image", image_to_display)
    # cv2.waitKey()
    
    rect1 = image[min_y1:max_y1, min_x1:max_x1]
    rect2 = image[min_y2:max_y2, min_x2:max_x2]
    rect3 = image[min_y3:max_y3, min_x3:max_x3]
    
    # cv2.imshow("Rectangle", rect1)
    # cv2.waitKey()
    # Convert the image from BGR to RGB
    rect1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2RGB)
    # Make a PIL image
    rect1 = Image.fromarray(rect1)
    # Save the image in the correct directory
    # rect1.save("rect1.png", "PNG")
    
    # cv2.imshow("Rectangle", rect2)
    # cv2.waitKey()
    # Convert the image from BGR to RGB
    rect2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2RGB)
    # Make a PIL image
    rect2 = Image.fromarray(rect2)
    # Save the image in the correct directory
    # rect2.save("rect2.png", "PNG")
    
    # cv2.imshow("Rectangle", rect3)
    # cv2.waitKey()
    # Convert the image from BGR to RGB
    rect3 = cv2.cvtColor(rect3, cv2.COLOR_BGR2RGB)
    # Make a PIL image
    rect3 = Image.fromarray(rect3)
    # Save the image in the correct directory
    # rect3.save("rect3.png", "PNG")
    
    rect1 = transforms.Compose([
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
    ])(rect1)

    # im1 = transforms.ToPILImage()(rect1)
    # plt.imshow(im1)
    # plt.show()

    rect2 = transforms.Compose([
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
    ])(rect2)
    
    rect3 = transforms.Compose([
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
    ])(rect3)
    
    rect1 = torch.unsqueeze(rect1, dim=0)
    rect1 = rect1.to(device)
    cost1 = model(rect1).item()
    
    rect2 = torch.unsqueeze(rect2, dim=0)
    rect2 = rect2.to(device)
    cost2 = model(rect2).item()
    
    rect3 = torch.unsqueeze(rect3, dim=0)
    rect3 = rect3.to(device)
    cost3 = model(rect3).item()
    
    return cost1, cost2, cost3
    
def display_trajectory(image, trajectory):
    points_world = np.zeros((trajectory.shape[0], 3))
    points_world[:, :2] = trajectory[:, :2]
    theta = trajectory[:, 2]

    # Create arrays to store left and write front wheels positions
    points_left_world = np.copy(points_world)
    points_right_world = np.copy(points_world)

    # Compute the distances between the wheels and the robot's origin
    delta_X = L*np.sin(theta)/2
    delta_Y = L*np.cos(theta)/2

    # Compute the positions of the outer points of the two
    # front wheels
    points_left_world[:, 0] -= delta_X
    points_left_world[:, 1] += delta_Y
    points_right_world[:, 0] += delta_X
    points_right_world[:, 1] -= delta_Y

    # Gather front wheels outer points coordinates in a single array
    points_world = np.concatenate([points_left_world,
                                   points_right_world])

    # Compute the points coordinates in the robot frame
    points_robot = apply_rigid_motion(points_world,
                                      ROBOT_TO_WORLD)

    # Compute the points coordinates in the camera frame
    points_camera = apply_rigid_motion(points_robot,
                                       CAM_TO_ROBOT)

    # Compute the points coordinates in the image plan
    points_image = camera_frame_to_image(points_camera,
                                         K)

    
    image_to_display = np.copy(image)
    
    points_image = points_image[::5]
    
    
    points_quadri1 = np.vstack([points_image[5:7], points_image[-2:]])
    min_x1 = np.int32(np.min(points_quadri1[:, 0], axis=0))
    max_x1 = np.int32(np.max(points_quadri1[:, 0], axis=0))
    min_y1 = np.int32(np.min(points_quadri1[:, 1], axis=0))
    max_y1 = np.int32(np.max(points_quadri1[:, 1], axis=0))
    
    points_quadri2 = np.vstack([points_image[4:6], points_image[-3:-1]])
    min_x2 = np.int32(np.min(points_quadri2[:, 0], axis=0))
    max_x2 = np.int32(np.max(points_quadri2[:, 0], axis=0))
    min_y2 = np.int32(np.min(points_quadri2[:, 1], axis=0))
    max_y2 = np.int32(np.max(points_quadri2[:, 1], axis=0))
    
    points_quadri3 = np.vstack([points_image[3:5], points_image[-4:-2]])
    min_x3 = np.int32(np.min(points_quadri3[:, 0], axis=0))
    max_x3 = np.int32(np.max(points_quadri3[:, 0], axis=0))
    min_y3 = np.int32(np.min(points_quadri3[:, 1], axis=0))
    max_y3 = np.int32(np.max(points_quadri3[:, 1], axis=0))
    
    image_to_display = draw_quadrilateral(image_to_display,
                                          np.array([[min_x1, max_y1],
                                                    [min_x1, min_y1],
                                                    [max_x1, min_y1],
                                                    [max_x1, max_y1]]),
                                          color=(255, 0, 0))
    image_to_display = draw_quadrilateral(image_to_display,
                                          np.array([[min_x2, max_y2],
                                                    [min_x2, min_y2],
                                                    [max_x2, min_y2],
                                                    [max_x2, max_y2]]),
                                          color=(255, 0, 0))
    image_to_display = draw_quadrilateral(image_to_display,
                                          np.array([[min_x3, max_y3],
                                                    [min_x3, min_y3],
                                                    [max_x3, min_y3],
                                                    [max_x3, max_y3]]),
                                          color=(255, 0, 0))
    
    # Draw the points on the image
    # image_to_display = draw_points(np.copy(image), points_image)
    image_to_display = draw_points(image_to_display, points_image)

    cv2.imshow("Image", image_to_display)
    cv2.waitKey()



# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_path_grass.bag"

# Topics name
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"
DEPTH_MAP_TOPIC = "/zed_node/depth/depth_registered"

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
# K = np.array([[700, 0, 320],
#               [0, 700, 180],
#               [0, 0, 1]])
K = np.array([[700, 0, 640],
              [0, 700, 360],
              [0, 0, 1]])
# K = np.array([[528.055, 0, 636.19],
#               [0, 528.105, 361.451],
#               [0, 0, 1]])

# Compute the transform matrix between the world and the robot
WORLD_TO_ROBOT = np.eye(4)

# Compute the inverse transform
ROBOT_TO_WORLD = inverse_transform_matrix(WORLD_TO_ROBOT)


# Initialize the bridge between ROS and OpenCV images
bridge = cv_bridge.CvBridge()

# Open the bag file
bag = rosbag.Bag(FILE)

_, _, t_image = next(iter(bag.read_messages(topics=[IMAGE_TOPIC])))
_, msg_image, t_image = next(iter(bag.read_messages(topics=[IMAGE_TOPIC], start_time=t_image+rospy.Duration(160))))

# _, msg_depth, t_depth = next(iter(bag.read_messages(topics=[DEPTH_MAP_TOPIC], start_time=t_image+rospy.Duration(150))))

# depth_map = bridge.imgmsg_to_cv2(msg_depth, desired_encoding="passthrough")
# depth_map_copy = np.copy(depth_map)
# print(depth_map_copy)
# depth_map_copy[np.isnan(depth_map)] = 0.
# depth_map_copy[depth_map_copy == np.inf] = 0
# print(depth_map_copy)

# depth_map_normalized = cv2.normalize(depth_map_copy, None, 0, 1., cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# print(depth_map_normalized)
# cv2.imshow("Normalized depth image", depth_map_normalized)
# cv2.waitKey()

# Convert the current ROS image to the OpenCV type
image = bridge.imgmsg_to_cv2(msg_image, desired_encoding="passthrough")


# Initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
x_init = [0., 0., 0., 0., 0.]


# Use a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# Load the pre-trained AlexNet model
model = models.resnet18().to(device=device)

# Replace the last layer by a fully-connected one with 1 output
model.fc = nn.Linear(model.fc.in_features, 1, device=device)

model.load_state_dict(torch.load("src/models_development/resnet18_fine_tuned_small_bag.params"))

model.eval()


for y in [-0.3, 0., 0.3]:
    
    trajectory = predict_trajectory(x_init, v=1., y=y)
    
    cost1, cost2, cost3 = eval_trajectory(image, trajectory)
    print(f"Trajectory cost: {cost1+cost2+cost3} ({cost1}+{cost2}+{cost3})\n")
    
    display_trajectory(image, trajectory)
