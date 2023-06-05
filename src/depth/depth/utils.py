import numpy as np
import cv2


class Depth():
    """
    A class to handle depth images. It can be used to compute the surface
    normals from a depth image, and to display the depth and normal images.
    """
    
    def __init__(self, depth: np.ndarray=None) -> None:
        """Constructor of the Depth class.

        Args:
            depth (np.ndarray, optional): A depth image.
            Defaults to None.
        """
        # Convert inf values to nan, in order to make numpy functions work
        depth[np.isinf(depth)] = np.nan
        
        # Set the depth image attribute
        self.depth_ = depth
        
        # Set the normal image attribute to None
        self.normal_ = None
        
        # Set a lambda function to convert the range of the depth values
        # to [0, 255] (the minimum depth value is supposed to be 0)
        self.convert_range_depth =\
            lambda depth: np.uint8((255/np.max(depth))*depth)
        
        # Set a lambda function to convert the range of the normal components
        # to [0, 255] (each component is supposed to be in [-1, 1])
        self.convert_range_normal =\
            lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
        
        # Set a lambda function to fill the missing values in the depth image
        self.fill_missing_depth =\
            lambda depth, default_value=-1: np.where(np.isfinite(depth),
                                                     depth, default_value)
        
        # Set a lambda function to fill the missing values in the normal image
        self.fill_missing_normal =\
            lambda normal, default_normal=[0,0,1]: np.where(
                np.isfinite(normal).all(2, keepdims=True),
                normal,
                default_normal)
    
    
    def compute_normals(self,
                        K: np.ndarray,
                        bilateral_filter: dict=None) -> None:
        """Compute the surface normals from a depth image.

        Args:
            K (np.ndarray): The internal calibration matrix of the camera.
            bilateral_filter (dict, optional): The parameters of the bilateral
            filter. Defaults to None.

        Raises:
            ValueError: If the depth image has not been set.
        """
        # Raise an error if the depth map has not been set
        if self.depth_ is None:
            raise ValueError("The depth image is empty. "
                             "Please set it before trying to use it.")
        
        # Apply a bilateral filter to smooth the image and reduce noise
        depth = cv2.bilateralFilter(self.depth_, **bilateral_filter)\
                if bilateral_filter is not None else self.depth_
    
        # Get the focal lengths from the internal calibration matrix
        fx, fy = K[0][0], K[1][1]
        
        # Compute the gradients of the depth image
        # u, v are the pixel coordinates in the image
        dz_dv, dz_du = np.gradient(depth)
        
        # Derive the pinhole camera model
        # x, y, z are the coordinates in the camera coordinate system
        du_dx = fx / depth
        dv_dy = fy / depth

        # Apply the chain rule for the partial derivatives
        dz_dx = dz_du * du_dx
        dz_dy = dz_dv * dv_dy

        # Compute the surface normals from the gradients with a cross-product
        # (1, 0, dz_dx) X (0, 1, dz_dy) = (-dz_dx, -dz_dy, 1)
        normal = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))

        # Normalize to unit vector
        normal = normal / np.linalg.norm(normal,
                                         axis=2,
                                         keepdims=True)

        # Set the normal attribute
        self.normal_ = normal
    
    
    def display_depth(self) -> None:
        """Display the depth image.

        Raises:
            ValueError: If the depth image has not been set.
        """
        # Raise an error is the depth image has not been set
        if self.depth_ is None:
            raise ValueError("The depth image is empty. "
                             "Please set it before trying to display it.")
        
        # Display the depth image after filling the missing values
        # and converting the range of the values
        cv2.imshow("Depth image",
                   self.convert_range_depth(
                       self.fill_missing_depth(self.depth_)))
    
    
    def display_normal(self) -> None:
        """Display the normal map.

        Raises:
            ValueError: If the normal map has not been set.
        """        
        # Raise an error is the normal map has not been set
        if self.normal_ is None:
            raise ValueError("The normal map attribute is empty. "
                             "Please compute it before trying to display it.")
        
        # Display the normal map after filling the missing values
        # and converting the range of the values
        cv2.imshow("Surface normals",
                   self.convert_range_normal(
                       self.fill_missing_normal(self.normal_)))
    
    
    def get_depth(self, fill: bool=True) -> np.ndarray:
        """Getter of the depth image attribute.

        Returns:
            np.ndarray: The depth image.
        """
        return self.fill_missing_depth(self.depth_) if fill else self.depth_

    
    def set_depth(self, depth: np.ndarray) -> None:
        """Setter of the depth image attribute.

        Args:
            depth (np.ndarray): The depth image.
        """
        self.depth_ = depth


    def get_normal(self, fill: bool=True) -> np.ndarray:
        """Getter of the normal image attribute.

        Returns:
            np.ndarray: The normal image.
        """        
        return self.fill_missing_normal(self.normal_) if fill else self.normal_


    def set_normal(self, normal: np.ndarray) -> None:
        """Setter of the normal image attribute.

        Args:
            normal (np.ndarray): The normal image.
        """        
        self.normal_ = normal
    