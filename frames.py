import numpy as np
import tf.transformations


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