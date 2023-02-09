import cv2
import numpy as np


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