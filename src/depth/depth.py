import rosbag
import numpy as np
import cv2
import cv_bridge
import matplotlib.pyplot as plt


def compute_normals(depth_map):
    # First, compute the gradient of the depth map using Sobel filters
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    
    grad_x = convert_range(grad_x, np.min(grad_x), np.max(grad_x), 0, 255).astype(np.uint8)
    grad_y = convert_range(grad_y, np.min(grad_y), np.max(grad_y), 0, 255).astype(np.uint8)
    cv2.imshow("Grad x", grad_x)
    cv2.imshow("Grad y", grad_y)
    
    cv2.waitKey()
    
    print(np.max(grad_x))

    # Next, compute the cross product of the gradient vectors to get the surface normals
    normals = np.zeros((depth_map.shape[0], depth_map.shape[1], 3))
    normals[:,:,0] = -grad_x / np.sqrt(grad_x**2 + grad_y**2 + 1)
    normals[:,:,1] = -grad_y / np.sqrt(grad_x**2 + grad_y**2 + 1)
    normals[:,:,2] = 1 / np.sqrt(grad_x**2 + grad_y**2 + 1)
    
    return normals

def compute_normals_2(depth_image):
    # Convert depth image to float32
    depth_image = depth_image.astype(np.float32)

    # Compute gradient in X and Y directions using Sobel operator
    grad_x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1)

    # Compute surface normals using cross product
    norm_x = -grad_x / np.sqrt(grad_x ** 2 + grad_y ** 2 + 1)
    norm_y = -grad_y / np.sqrt(grad_x ** 2 + grad_y ** 2 + 1)
    norm_z = 1 / np.sqrt(grad_x ** 2 + grad_y ** 2 + 1)
    
    norm_x = convert_range(norm_x, -1, 1, 0, 255).astype(np.uint8)
    norm_y = convert_range(norm_y, -1, 1, 0, 255).astype(np.uint8)
    norm_z = convert_range(norm_z, -1, 1, 0, 255).astype(np.uint8)

    # Normalize the surface normals
    normals = np.dstack((norm_x, norm_y, norm_z))
    # return cv2.normalize(normals, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
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


# depth_image = cv2.imread("depth_image_test.png", cv2.IMREAD_GRAYSCALE)
# print(type(depth_image[0,0]))
# normals = compute_normals(depth_image)

# cv2.imshow("Normals", normals)
# cv2.waitKey()


bridge = cv_bridge.CvBridge()

# Open the bag file
# bag = rosbag.Bag("bagfiles/raw_bagfiles/simulation.bag")
bag = rosbag.Bag("bagfiles/raw_bagfiles/depth.bag")

topics = [
    "/zed_node/depth/depth_registered",
    # "/camera1/image_raw_depth",
    "/imu/data",
]

# Loop through the messages in the bag file
# for topic, msg, t in bag.read_messages(topics=topics):
# for topic, msg, t in bag.read_messages(topics=["/camera1/image_raw_depth"]):
for topic, msg, t in bag.read_messages(topics=["/zed_node/depth/depth_registered"]):
    
    # Convert the ROS Image to an opencv image
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    
    # Make a copy of the depth image (the original one is read-only)
    depth_image = np.copy(depth_image)
    
    # depth_image = cv2.imread("src/depth/depth_image_test.png", cv2.IMREAD_GRAYSCALE)
    
    # cv2.imwrite("depth_image_zed.png", depth_image)
    
    # print(np.min(depth_image[np.isnan(depth_image) == False]))
    # break
    
    # Replace NaN and Inf values (missing information) by a default value
    index_holes = (np.isnan(depth_image)) | (np.isinf(np.abs(depth_image)))
    
    default_value = 7.
    
    depth_image[index_holes] = default_value
    
    # Change the range of the depth values
    depth_image = convert_range(depth_image, 0.7, 7, 0, 255).astype(np.uint8)
    
    cv2.imwrite("depth_image_zed.png", depth_image)
    break

    
    # grad_x = cv2.Sobel(depth_image, cv2.CV_32F , 1, 0)
    # grad_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1)
    
    # print(grad_x)
    
    # print(np.min(depth_image), np.max(depth_image))
    
    # normals = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
    
    # for x in range(1, depth_image.shape[0]-1):
    #     for y in range(1, depth_image.shape[1]-1):
            
    #         dzdx = (depth_image[x+1, y] - depth_image[x-1, y])/2
    #         dzdy = (depth_image[x, y+1] - depth_image[x, y-1])/2
            
    #         d = np.array([-dzdx,
    #                       -dzdy,
    #                       1])
            
    #         norm = np.linalg.norm(d)
            
    #         normals[x, y] = d/norm
            
            # if x%20 == 0:
            #     print(d/norm)
            
    
    # normals = normals[1:-1, 1:-1]
            
    
    # Next, compute the cross product of the gradient vectors to get the surface normals
    # normals[:,:,0] = -grad_x
    # normals[:,:,1] = -grad_y
    # normals[:,:,2] = 1
    
    # print(normals.shape)
    # print(np.linalg.norm(normals, axis=2).shape)
    
    # print(normals[100, 100])
     
    # normals[:, :, 0] = convert_range(normals[:, :, 0], -1., 1., 0, 255).astype(np.uint8)
    # normals[:, :, 1] = convert_range(normals[:, :, 1], -1., 1., 0, 255).astype(np.uint8)
    # normals[:, :, 2] = convert_range(normals[:, :, 2], -1., 1., 0, 255).astype(np.uint8)
    
    # print(normals[100, 100])
    
    # print(normals)
    
    # print(np.min(np.linalg.norm(normals, axis=2)), np.max(np.linalg.norm(normals, axis=2)))
    # normals /= np.sqrt(grad_x**2 + grad_y**2 + 1)
    
    # cv2.imshow("normals", normals)
    
    
    # print(np.min(grad_x), np.max(grad_x))
    # print(np.min(grad_x[~index_holes]), np.max(grad_x[~index_holes]))
    
    # grad_x = convert_range(grad_x, np.min(grad_x[~index_holes]), np.max(grad_x[index_holes]), 0, 255).astype(np.uint8)
    
    # grad_x[index_holes] = 0
    
    # grad_x = cv2.applyColorMap(grad_x, cv2.COLORMAP_JET)
    
    # cv2.imshow("Gradient x", grad_x)
    
    # cv2.imwrite("depth_image_zed.tiff", depth_image)
    # break
    
    # Apply bilateral filtering
    # depth_image = cv2.bilateralFilter(depth_image, 5, 75, 75)
    
    # print(depth_image)
    # break
    
    # print(np.max(depth_image))
    # break
    
    # # Convert to float32
    # depth_image = cv2.convertScaleAbs(depth_image, alpha=1/256)
    
    # # Normalize the pixel values
    # depth_image_norm = depth_image / (np.max(depth_image) + 0.01)

    # # Rescale to 0-255 and convert to integer
    # depth_image_gray = cv2.convertScaleAbs(depth_image_norm*255)
    
    
    # depth_image_normalized = cv2.normalize(depth_image,
    #                                        None,
    #                                        0,
    #                                        255,
    #                                        cv2.NORM_MINMAX,
    #                                        dtype=cv2.CV_32F)
    
    # Set camera parameters
    # K = [[534, 0, 634],
    #      [0, 534, 363],
    #      [0, 0, 1]]

    # Compute normals
    # rgbd_normals = cv2.RgbdNormals(depth_image_normalized.shape[0],
    #                                depth_image_normalized.shape[1],
    #                                cv2.CV_32F,
    #                                K,
    #                                window_size=5)
    # normals = rgbd_normals(depth_image_normalized)
    
    # normals = compute_normals_2(depth_image)
    
    cv2.imshow("Depth map", depth_image)
    cv2.waitKey()
    # print(topic)
    
    # break
    
# Close the bag file
bag.close()
