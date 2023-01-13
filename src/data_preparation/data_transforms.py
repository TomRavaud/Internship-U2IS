"""
Illustration of images transforms
"""

# Import libraries and modules
from PIL import Image
import matplotlib.pyplot as plt

# Provide support form images use
from torchvision import transforms


# Code executed when the file runs as a script,
# but not when imported as a module
if __name__ == "__main__":
    # Open the first image of the sample dataset
    image = Image.open(
        "datasets/dataset_sample_bag/zed_node_rgb_image_rect_color/00000.png")
    
    # Create a new figure
    plt.figure()

    # Display the original image
    plt.subplot(221)
    plt.imshow(image)
    plt.title("Original image")
    
    # Resize the image
    image_resized = transforms.Resize(300)(image)
    
    # Display the resized image
    plt.subplot(222)
    plt.imshow(image_resized)
    plt.title("Resized image")
    
    # Crop the image at the center
    image_cropped = transforms.CenterCrop(200)(image)
    
    # Display the cropped image
    plt.subplot(223)
    plt.imshow(image_cropped)
    plt.title("Cropped image")
    
    # Random horizontal flip
    image_flipped = transforms.RandomHorizontalFlip(p=1)(image)
    
    # Display the flipped image
    plt.subplot(224)
    plt.imshow(image_flipped)
    plt.title("Flipped image")
    
    plt.show()
    