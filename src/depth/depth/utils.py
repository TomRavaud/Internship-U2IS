import numpy as np
import cv2


class Depth():
    """
    A class to handle depth images. It can be used to compute the surface
    normals from a depth image, and to display the depth and normal images.
    """
    
    def __init__(self,
                 depth: np.ndarray=None,
                 depth_range: tuple=None) -> None:
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
            lambda depth: np.uint8((255/np.max(depth))*depth)\
            if depth_range is None\
            else np.uint8(255/(depth_range[1]-depth_range[0])*(depth-depth_range[0]))
        
        # Set a lambda function to convert the range of the normal components
        # to [0, 255] (each component is supposed to be in [-1, 1])
        self.convert_range_normal =\
            lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
        
        # Set a lambda function to fill the missing values in the depth image
        self.fill_missing_depth =\
            lambda depth, default_value: np.where(np.isfinite(depth),
                                                     depth, default_value)
        
        # Set a lambda function to fill the missing values in the normal image
        self.fill_missing_normal =\
            lambda normal, default_normal: np.where(
                np.isfinite(normal).all(2, keepdims=True),
                normal,
                default_normal)
    
    
    def compute_normal(self,
                       K: np.ndarray,
                       bilateral_filter: dict=None,
                       gradient_threshold: float=None) -> None:
        """Compute the surface normals from a depth image.

        Args:
            K (np.ndarray): The internal calibration matrix of the camera.
            bilateral_filter (dict, optional): The parameters of the bilateral
            filter. Defaults to None.
            gradient_threshold (float, optional): The threshold for the
            gradient magnitude. If the gradient magnitude is above this
            threshold, the normal is set to nan. Defaults to None.

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
        
        # Set the normal to nan if the gradient magnitude is above a threshold
        # (this allows to remove the noise at edges of objects)
        if gradient_threshold is not None:
            
            # Compute the gradient magnitude
            gradient_magnitude = np.linalg.norm(np.dstack((dz_dx, dz_dy)),
                                                axis=2,
                                                keepdims=True)
            
            # Set the normal to nan if the gradient magnitude is above a
            # threshold
            normal = np.where(gradient_magnitude < gradient_threshold,
                              normal,
                              np.full_like(normal, np.nan))
            
        # Set the normal attribute
        self.normal_ = normal
        
    
    def display_depth(self, name: str="Depth image") -> None:
        """Display the depth image.

        Args:
            name (str, optional): Name of the window.
            Defaults to "Depth image".
        Raises:
            ValueError: If the depth image has not been set.
        """
        # Raise an error is the depth image has not been set
        if self.depth_ is None:
            raise ValueError("The depth image is empty. "
                             "Please set it before trying to display it.")
        
        # Display the depth image after filling the missing values
        # and converting the range of the values
        cv2.imshow(name,
                   self.convert_range_depth(
                       self.fill_missing_depth(self.depth_, 0.7)))
        
    
    def display_normal(self, name: str="Surface normals") -> None:
        """Display the normal map.
        
        Args:
            name (str, optional): Name of the window.
            Defaults to "Surface normals".

        Raises:
            ValueError: If the normal map has not been set.
        """        
        # Raise an error is the normal map has not been set
        if self.normal_ is None:
            raise ValueError("The normal map attribute is empty. "
                             "Please compute it before trying to display it.")
        
        # Display the normal map after filling the missing values
        # and converting the range of the values
        cv2.imshow(name,
                   self.convert_range_normal(
                       self.fill_missing_normal(self.normal_, [0, 0, 1])))
    
    
    def get_depth(self,
                  fill: bool=True,
                  default_depth: float=0.7,
                  convert_range: bool=True) -> np.ndarray:
        """Getter of the depth image attribute.
        
        Args:
            fill (bool, optional): Whether to fill the missing values.
            Defaults to True.
            default_depth (float, optional): The value to use to fill the
            missing values. Defaults to -1.
            convert_range (bool, optional): Whether to convert the range of
            the values. Defaults to True.

        Returns:
            np.ndarray: The depth image.
        """
        depth = self.depth_
        
        # Fill the missing values if required
        if fill:
            depth = self.fill_missing_depth(depth, default_depth)
        
        # Convert the range of the values if required
        if convert_range:
            depth = self.convert_range_depth(depth)
            
        return depth

    
    def set_depth(self, depth: np.ndarray) -> None:
        """Setter of the depth image attribute.

        Args:
            depth (np.ndarray): The depth image.
        """
        self.depth_ = depth


    def get_normal(self,
                   fill: bool=True,
                   default_normal: list=[0, 0, 1],
                   convert_range: bool=True) -> np.ndarray:
        """Getter of the normal image attribute.
        
        Args:
            fill (bool, optional): Whether to fill the missing values.
            Defaults to True.
            default_normal (list, optional): The value to use to fill the
            missing values. Defaults to [0, 0, 1].
            convert_range (bool, optional): Whether to convert the range of
            the values. Defaults to True.

        Returns:
            np.ndarray: The normal image.
        """
        normal = self.normal_
        
        # Fill the missing values if required
        if fill:
            normal = self.fill_missing_normal(normal, default_normal)
        
        # Convert the range of the values if required
        if convert_range:
            normal = self.convert_range_normal(normal)

        return normal


    def set_normal(self, normal: np.ndarray) -> None:
        """Setter of the normal image attribute.

        Args:
            normal (np.ndarray): The normal image.
        """        
        self.normal_ = normal
