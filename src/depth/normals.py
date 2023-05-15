import numpy as np
import cv2
from scipy.interpolate import griddata

import rosbag
import cv_bridge



def compute_normals(depth_map):
    # First, compute the gradient of the depth map using Sobel filters
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

    # Next, compute the cross product of the gradient vectors to get the surface normals
    normals = np.zeros((depth_map.shape[0], depth_map.shape[1], 3))
    normals[:,:,0] = -grad_x / np.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    normals[:,:,1] = -grad_y / np.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    normals[:,:,2] = 1 / np.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    return normals


IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"
DEPTH_MAP_TOPIC = "/zed_node/depth/depth_registered"

# Initialize the bridge between ROS and OpenCV images
bridge = cv_bridge.CvBridge()

bag = rosbag.Bag("bagfiles/raw_bagfiles/tom_1.bag")

# _, _, t_image = next(iter(bag.read_messages(topics=[IMAGE_TOPIC])))
# _, msg_image, t_image = next(iter(bag.read_messages(topics=[IMAGE_TOPIC], start_time=t_image+rospy.Duration(100))))

_, msg_depth, t_depth = next(iter(bag.read_messages(topics=[DEPTH_MAP_TOPIC])))

# Convert the depth map to an OpenCV image
depth_map_raw = bridge.imgmsg_to_cv2(msg_depth, desired_encoding="passthrough")

# Make a copy of the depth map because it is read only
depth_map = np.copy(depth_map_raw)

# Define the desired output size
output_size = (1280, 720)

# Compute the scaling factor for each dimension
scale_x = output_size[0] / depth_map.shape[1]
scale_y = output_size[1] / depth_map.shape[0]

# Scale the depth map using bilinear interpolation
scaled_depth_map = cv2.resize(depth_map, output_size, interpolation=cv2.INTER_LINEAR)


# # Find the valid (non-NaN) depth values
# valid_mask = ~np.isnan(depth_map)
# valid_points = np.argwhere(valid_mask)
# valid_depths = depth_map[valid_mask]

# # Pad the depth map with valid values on the borders
# border_size = 10
# padded_depth_map = np.pad(depth_map, ((border_size, border_size), (border_size, border_size)), mode='edge')

# # Create a grid of pixel coordinates for the padded depth map
# padded_grid_x, padded_grid_y = np.meshgrid(np.arange(128 + 2*border_size), np.arange(72 + 2*border_size))

# # Interpolate the missing depth values using the valid points and the padded grid
# interpolated_depths = griddata(valid_points + border_size, valid_depths, (padded_grid_x, padded_grid_y), method='linear')

# # Crop the interpolated depth map to remove the padded borders
# cropped_interpolated_depths = interpolated_depths[border_size:-border_size, border_size:-border_size]

# # Replace the NaN values in the original depth map with the interpolated values
# depth_map[np.isnan(depth_map)] = cropped_interpolated_depths[np.isnan(depth_map)]

# print(depth_map)


# depth_map[np.isnan(depth_map)] = 0.
# print(np.count_nonzero(depth_map), 72*128)

# depth_map_copy = np.copy(depth_map)
# print(depth_map_copy)
# depth_map_copy[np.isnan(depth_map)] = 0.
# depth_map_copy[depth_map_copy == np.inf] = 0
# print(depth_map_copy)

depth_map_normalized = cv2.normalize(scaled_depth_map, None, 0, 1., cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# print(depth_map_normalized)

# Display the scaled depth maps
cv2.imshow('Scaled depth map', scaled_depth_map)
cv2.waitKey()
