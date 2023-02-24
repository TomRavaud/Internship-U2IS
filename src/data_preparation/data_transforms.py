"""
Illustration of images transforms
"""

# Import libraries and modules
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Provide support form images use
from torchvision import transforms


# Code executed when the file runs as a script,
# but not when imported as a module
if __name__ == "__main__":
    # Open the first image of the sample dataset
    image = Image.open(
        "datasets/dataset_tom_path_grass/zed_node_rgb_image_rect_color/00000.png")
    # image = Image.open(
    #     "datasets/dataset_sample_bag/zed_node_rgb_image_rect_color/00000.png")
    
    # Create a new figure
    plt.figure()

    # Display the original image
    plt.subplot(331)
    plt.imshow(image)
    plt.title("Original image")
    
    # Resize the image
    image_resized = transforms.Resize(200)(image)
    
    # Display the resized image
    plt.subplot(332)
    plt.imshow(image_resized)
    plt.title("Resized image")
    
    # Crop the image at the center
    image_cropped = transforms.CenterCrop(100)(image)
    
    # Display the cropped image
    plt.subplot(333)
    plt.imshow(image_cropped)
    plt.title("Cropped image")
    
    # Random horizontal flip
    image_flipped = transforms.RandomHorizontalFlip(p=1)(image)
    
    # Display the flipped image
    plt.subplot(334)
    plt.imshow(image_flipped)
    plt.title("Flipped image")
    
    
    # Normalize the image (in fact tensor) (mean and standard deviation are
    # pre-computed on the ImageNet dataset)
    tensor_normalized = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
        ])(image)
    
    # Convert the tensor to a PIL Image
    image_normalized = transforms.ToPILImage()(tensor_normalized)
    
    # Display the normalized image
    plt.subplot(335)
    plt.imshow(image_normalized)
    plt.title("Normalized image")
    
    
    # De-normalize the normalized tensor
    tensor_denormalized = transforms.Compose([
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1/0.229, 1/0.224, 1/0.225]
            ),
        transforms.Normalize(
            mean=[-0.485, -0.456, -0.406],
            std=[1., 1., 1.]
            ),
        ])(tensor_normalized)
    
    # Convert the tensor to a PIL Image
    image_denormalized = transforms.ToPILImage()(tensor_denormalized)
    
    plt.subplot(336)
    plt.imshow(image_denormalized)
    plt.title("De-normalized image")
    
    
    # Modify the brightness and the contrast of the image
    image_brightness_modified = transforms.ColorJitter(contrast=0.5, brightness=0.5)(image)
    
    plt.subplot(337)
    plt.imshow(image_brightness_modified)
    plt.title("Modified brightness")
    
    
    # Add some random gaussian noise to the image
    image_noisy = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + (0.001**0.5)*torch.randn(x.shape)),
        transforms.ToPILImage()
    ])(image)
    
    plt.subplot(338)
    plt.imshow(image_noisy)
    plt.title("Noisy image")
    
    plt.show()
    