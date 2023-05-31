import numpy as np
import cv2


def fill_nan_inf(image, default_value=-1.):
    """Replace nan and inf values in a depth image by a default value.

    Args:
        image (ndarray): The image to fill.
        default_value (float, optional): The value used to fill the holes.
        Defaults to -1.

    Returns:
        ndarray: The image with no nan and inf values.
    """
    index_holes = (np.isnan(image)) | (np.isinf(np.abs(image)))
    image[index_holes] = default_value
    
    return image

def compute_normals(depth_image):
    """Compute the normals of a depth image.

    Args:
        depth_image (ndarray): The depth image to compute the normals from.

    Returns:
        ndarray: The normals of the depth image.
    """
    
    # mask = depth_image == -1.
    
    # Compute the gradients of the depth image
    # gradient_x = np.gradient(np.ma.array(depth_image, mask=mask), axis=1)
    # gradient_y = np.gradient(np.ma.array(depth_image, mask=mask), axis=0)
    # gradient_x = np.gradient(depth_image, axis=1)
    # gradient_y = np.gradient(depth_image, axis=0)
    
    filter_x = np.array([-1, 0, 1])
    filter_y = np.array([[-1], [0], [1]])
    
    
    # gradient_x = cv2.filter2D(depth_image, -1, filter_x)
    # gradient_y = cv2.filter2D(depth_image, -1, filter_y)
    
    gradient_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
    
    abs_gradient_x = np.abs(gradient_x)
    abs_gradient_y = np.abs(gradient_y)
    
    print(depth_image)
    print(abs_gradient_x)
    
    gradient_x = 10*np.uint8(abs_gradient_x)
    gradient_y = 10*np.uint8(abs_gradient_y)
    
    # kernel_x, kernel_y = cv2.getDerivKernels(1, 0, 3)
    # print(kernel_x, kernel_y)
    
    # print(gradient_x == gradient_y)
    
    cv2.imshow("gradient_x", gradient_x)
    cv2.imshow("gradient_y", gradient_y)
    cv2.waitKey(0)

    # Compute the normals
    normals = np.stack((-gradient_x, -gradient_y, np.ones_like(depth_image)), axis=-1)
    
    # print(normals)
    
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    
    # print(normals)

    return normals

def convert_range(x1, min1, max1, min2, max2):
    """Map a number from one range to another
    Args:
        x1 (float): A number in the first range
        min1 (float): Lower bound of the first range
        max1 (float): Upper bound of the first range
        min2 (float): Lower bound of the second range
        max2 (float): Upper bound of the second range
    Returns:
        float: Mapped number in the second range
    """
    A = (max2-min2) / (max1-min1)
    B = (min2*max1 - min1*max2) / (max1 - min1)
    x2 = A*x1 + B
    
    return x2
