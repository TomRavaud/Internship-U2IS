import numpy as np
import cv2
import rosbag
import tf.transformations
import cv_bridge
import rospy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


def draw_grid(image):
    h, w, _ = image.shape
    dy, dx = 70, 210
    rows, cols = int(h/2) // dy, w // dx
    
    # Draw vertical lines
    for x in range(cols+1):
        cv2.line(image, (x*dx, int(h/2)), (x*dx, h), (255, 0, 0))

    # Draw horizontal lines
    for y in range(rows+1):
        cv2.line(image, (0, h-y*dy), (w, h-y*dy), (255, 0, 0)) 
    
    return image

def fill_cell(image, x, y, color=(0, 0, 255), transparency=0.3):
    h = image.shape[0]
    dy, dx = 70, 210
    
    overlay = np.copy(image)
    
    cv2.rectangle(overlay, (x*dx, h-y*dy), ((x+1)*dx, h-(y+1)*dy), color, -1)
    
    image = cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0)
    
    return image

def linear_gradient(cost, cost_min, cost_max):
    green = 255/(cost_min - cost_max)*cost + 255*cost_max/(cost_max - cost_min)
    red = 255 - green
    
    return (0, green, red)
    
def predict_cell_cost(image, x, y, model, transform):
    h = image.shape[0]
    dy, dx = 70, 210
    
    # Get the cell region in the image
    region_to_predict = image[h-(y+1)*dy:h-y*dy,
                              x*dx:(x+1)*dx]
    
    # Convert the image from BGR to RGB
    region_to_predict = cv2.cvtColor(region_to_predict, cv2.COLOR_BGR2RGB)
    # Make a PIL image
    region_to_predict = Image.fromarray(region_to_predict)
    
    # Apply transforms to the image
    region_to_predict = transform(region_to_predict)
    
    # Add a dimension of size one (to create a batch of size one)
    region_to_predict = torch.unsqueeze(region_to_predict, dim=0)
    region_to_predict = region_to_predict.to(device)
    cost = model(region_to_predict).item()
    
    return cost
    
def display_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey()

def compute_costmap(image, model, transform):
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Make a PIL image
    image = Image.fromarray(image)
    
    # Apply transforms to the image
    image = transform(image)
    
    # Add a dimension of size one (to create a batch of size one)
    image = torch.unsqueeze(image, dim=0)
    image = image.to(device)
    cost = model(image).item()
    
    print(cost)
    
    return

def display_costmap():
    return


# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_path_grass.bag"

# Topics name
IMAGE_TOPIC = "/zed_node/rgb/image_rect_color"

# Initialize the bridge between ROS and OpenCV images
bridge = cv_bridge.CvBridge()

# Open the bag file
bag = rosbag.Bag(FILE)

# Get an image from the bag file
_, _, t_image = next(iter(bag.read_messages(topics=[IMAGE_TOPIC])))
_, msg_image, t_image = next(iter(bag.read_messages(topics=[IMAGE_TOPIC], start_time=t_image+rospy.Duration(100))))

# Convert the current ROS image to the OpenCV type
image = bridge.imgmsg_to_cv2(msg_image, desired_encoding="passthrough")

# Use a GPU is available
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# Load the pre-trained model
model = models.resnet18().to(device=device)

# Replace the last layer by a fully-connected one with 1 output
model.fc = nn.Linear(model.fc.in_features, 1, device=device)

# Load the fine-tuned weights
model.load_state_dict(torch.load("src/models_development/models_parameters/resnet18_fine_tuned_small_bag.params"))

model.eval()

transform = transforms.Compose([ 
    # transforms.Resize(100),
    transforms.Resize((70, 210)),
    # transforms.Grayscale(),
    # transforms.CenterCrop(100),
    # transforms.RandomCrop(100),
    transforms.ToTensor(),
    # Mean and standard deviation were pre-computed on the training data
    # (on the ImageNet dataset)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Copy the image
image_to_display = np.copy(image)

for x in range(6):
    for y in range(5):
        # Compute the traversal cost in a given cell
        cell_cost = predict_cell_cost(image, x, y, model, transform)
        
        # print(cell_cost)

        # color = linear_gradient(cell_cost, 28, 250)
        color = linear_gradient(cell_cost, 4, 32)

        # Fill a given cell in the grid
        image_to_display = fill_cell(image_to_display, x, y, color)

# Draw a grid on the image
image_to_display = draw_grid(image_to_display)

# Display the resulting image
display_image(image_to_display)
