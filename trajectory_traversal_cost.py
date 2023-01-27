import numpy as np
import cv2
import rosbag
import tf.transformations
import cv_bridge


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


# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_road.bag"

# Topics name
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"

# Initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
x_init = [0., 0., 0., 0., 0.]

trajectory = predict_trajectory(x_init, v=1., y=0.3)



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

# Compute the transform matrix between the world and the robot
WORLD_TO_ROBOT = np.eye(4)

# Compute the inverse transform
ROBOT_TO_WORLD = inverse_transform_matrix(WORLD_TO_ROBOT)


# Initialize the bridge between ROS and OpenCV images
bridge = cv_bridge.CvBridge()

# Open the bag file
bag = rosbag.Bag(FILE)

_, msg_image, t_image = next(iter(bag.read_messages(topics=[IMAGE_TOPIC])))

# Convert the current ROS image to the OpenCV type
image = bridge.imgmsg_to_cv2(msg_image, desired_encoding="passthrough")

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

# Draw the points on the image
image = draw_points(image, points_image)

cv2.imshow("Image", image)
cv2.waitKey()
