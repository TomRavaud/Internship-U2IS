import cv2
import numpy as np


depth_image = cv2.imread("WuManchu_0360.png", cv2.IMREAD_ANYDEPTH)

normals = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))

# print(depth_image)
# print(depth_image.shape)

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


for x in range(1, depth_image.shape[1]):
    for y in range(1, depth_image.shape[0]):
        
        A = np.array([x, y, depth_image[y, x]])
        B = np.array([x, y-1, depth_image[y-1, x]])
        C = np.array([x-1, y, depth_image[y, x-1]])
        
        AB = B - A
        AC = C - A
        
        normal = np.cross(AB, AC)
        
        normals[y, x] = normal/np.linalg.norm(normal)


# normals = (normals + 1)/2
normals = convert_range(normals, -1, 1., 0, 255).astype(np.uint8)

print(normals)

cv2.imshow("Normals", normals)
cv2.waitKey()
