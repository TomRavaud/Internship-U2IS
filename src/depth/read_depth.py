import cv2


depth_image = cv2.imread("depth_image_zed.tiff", cv2.IMREAD_ANYDEPTH)

print(depth_image)
# cv2.imshow("image", depth_image*255)
# cv2.waitKey()
