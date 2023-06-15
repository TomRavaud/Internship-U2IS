import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import custom modules and packages
import params.learning
from dataset import TraversabilityDataset


def compute_mean_std(images_directory: str,
                     traversal_costs_file: str) -> tuple:
    """Compute the mean and standard deviation of the images of the dataset

    Args:
        images_directory (string): Directory with all the images
        traversal_costs_file (string): Name of the csv file which contains
        images index and their associated traversal cost

    Returns:
        tuple: Mean and standard deviation of the dataset
    """
    transform = transforms.Compose([
        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),
        
        # Reduce the size of the images
        # (if size is an int, the smaller edge of the
        # image will be matched to this number and the ration is kept)
        transforms.Resize(params.learning.IMAGE_SHAPE),
    ])
     
    # Create a Dataset for training
    dataset = TraversabilityDataset(
        traversal_costs_file=params.learning.DATASET+traversal_costs_file,
        images_directory=params.learning.DATASET+images_directory,
        transform_image=transform,
        transform_depth=transform,
        transform_normal=transform
    )
     
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    cnt = 0
    first_moment = torch.empty(7)
    second_moment = torch.empty(7)

    for images, traversal_costs, traversability_labels, linear_velocity in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        first_moment = (cnt * first_moment + sum_) / (cnt + nb_pixels)
        second_moment = (cnt * second_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean = first_moment
    std = torch.sqrt(second_moment - first_moment ** 2)
    
    return mean, std
