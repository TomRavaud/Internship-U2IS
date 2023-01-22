"""
Step 2:

Prepare the data: it includes data loading, pre-processing and batching
(data are loaded from an external source, transformed, converted to the
appropriate format for model training (ie tensors) and batched)
"""

# Import modules and libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split


# Define the data to be used
DATASET = "datasets/dataset_sample_bag/"


class TraversabilityDataset(Dataset):
    """Custom Dataset class to represent our dataset
    It includes data and information about the data

    Args:
        Dataset (class): Abstract class which represents a dataset
    """
    
    def __init__(self, traversal_costs_file, images_directory,
                 transform=None):
        """Constructor of the class

        Args:
            traversal_costs_file (string): Path to the csv file which contains
            images index and their associated traversal cost
            images_directory (string): Directory with all the images
            transform (callable, optional): Transforms to be applied on a
            sample. Defaults to None.
        """
        # Read the csv file
        self.traversal_costs_frame = pd.read_csv(traversal_costs_file)
        
        # Initialize the name of the images directory
        self.images_directory = images_directory
        
        # Initialize the transforms
        self.transform = transform

    def __len__(self):
        """Return the size of the dataset

        Returns:
            int: Number of samples
        """
        return len(self.traversal_costs_frame)

    def __getitem__(self, idx):
        """Allow to access a sample by its index

        Args:
            idx (int): Index of a sample

        Returns:
            list: Sample at index idx
            ([image, traversal_cost])
        """
        # Get the image name at index idx
        image_name = os.path.join(self.images_directory,
                                  self.traversal_costs_frame.iloc[idx, 0])
        
        # Read the image
        image = Image.open(image_name)
        
        # Get the corresponding traversal cost
        traversal_cost = self.traversal_costs_frame.iloc[idx, 1:]
        traversal_cost = np.array(traversal_cost)
        traversal_cost = np.float32(traversal_cost)

        # Create the sample
        sample = [image, traversal_cost]

        # Eventually apply transforms to the image
        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample
 

# Compose several transforms together to be applied to training data
# (Note that transforms are not applied yet)
train_transform = transforms.Compose([
    # Reduce the size of the images
    # (if size is an int, the smaller edge of the
    # image will be matched to this number and the ration is kept)
    transforms.Resize(300),
    
    # Crop the image at the center
    # (if size is an int, a square crop is made)
    transforms.CenterCrop(100),
    
    # Convert the image to grayscale
    # transforms.Grayscale(num_output_channels=1),
    
    # Perform horizontal flip of the image with a probability of 0.5
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Convert a PIL Image or numpy.ndarray to tensor
    transforms.ToTensor(),
    
    # Normalize a tensor image with pre-computed mean and standard deviation
    # (based on the data used to train the model(s))
    # (be careful, it only works on torch.*Tensor)
    # transforms.Normalize(
    #     mean=[0., 0., 0.],
    #     std=[0., 0., 0.]
    # )
])

# Define a different set of transforms testing
# (for instance we do not need to flip the image)
test_transform = transforms.Compose([
    transforms.Resize(300),
    # transforms.Grayscale(),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
    
    # Mean and standard deviation were pre-computed on the training data
    # transforms.Normalize(
    #     mean=[0., 0., 0.],
    #     std=[0., 0., 0.]
    # )
])


# Create a Dataset instance for our training data
train_data = TraversabilityDataset(
    traversal_costs_file=DATASET+"imu_data.csv",
    images_directory=DATASET+"zed_node_rgb_image_rect_color",
    transform=train_transform
)

# Split our training dataset into a training dataset and a validation dataset
train_set, val_set = random_split(train_data, [0.8, 0.2])

# # Create a Dataset instance for our testing data
# test_set = TraversabilityDataset(
#     traversal_costs_file="???",
#     images_directory="???",
#     transform=test_transform
# )



# Combine a dataset and a sampler, and provide an iterable over the dataset
# (setting shuffle argument to true calls a RandomSampler, and avoids to
# have to create a Sampler object)
train_loader = DataLoader(
    train_set,
    batch_size=2,
    shuffle=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=2,
    shuffle=True,
)

# test_loader = DataLoader(
#     test_set,
#     batch_size=5,
#     shuffle=False,  # SequentialSampler
# )


# Code executed when the file runs as a script,
# but not when imported as a module
if __name__ == "__main__":
    print("Number of training samples :", len(train_data))
    print(len(train_set), "for training")
    print(len(val_set), "for validation")
    
    # Go through the batches and display images
    for batch in train_loader:
        for image in batch[0]:
            plt.imshow(transforms.ToPILImage()(image), cmap="gray")
            plt.show()
    
    # The next() function returns the next item from the iterator,
    # ie the first batch (we can access the second batch with another
    # next() call)
    # train_images, train_traversal_costs = next(iter(train_loader))
    # print("Image batch shape : ", train_images.size())
    # print("Traversal cost batch shape : ", train_traversal_costs.size())
    # plt.imshow(transforms.ToPILImage()(train_images[0]))
    # plt.show()

    # TODO: To check
    # # Image normalization
    # # Concatenate all the images of the dataset
    # images = torch.cat([train_sample[0].unsqueeze(0) for train_sample in train_set], dim=0)
    # # Compute the mean of the images
    # mean = torch.mean(images)
    # # Compute the standard deviation of the entire training set
    # std = torch.std(images)

    # print(std)
    # print(mean)