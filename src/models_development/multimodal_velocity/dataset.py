import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import tifffile
import numpy as np


class TraversabilityDataset(Dataset):
    """Custom Dataset class to represent our dataset
    It includes data and information about the data

    Args:
        Dataset (class): Abstract class which represents a dataset
    """
    def __init__(self,
                 traversal_costs_file: str,
                 images_directory: str,
                 modes: str="c",
                 transform_image: callable=None,
                 transform_depth: callable=None,
                 transform_normal: callable=None) -> None:
        """Constructor of the class

        Args:
            traversal_costs_file (string): Path to the csv file which contains
            images index and their associated traversal cost
            images_directory (string): Directory with all the images
            transform_image (callable, optional): Transforms to be applied on a
            rdg image. Defaults to None.
            transform_depth (callable, optional): Transforms to be applied on a
            depth image. Defaults to None.
            transform_normal (callable, optional): Transforms to be applied on a
            normal image. Defaults to None.
        """
        # Read the csv file
        self.traversal_costs_frame = pd.read_csv(traversal_costs_file,
                                                 converters={"image_id": str})
        
        # Initialize the name of the images directory
        self.images_directory = images_directory
        
        # Initialize the modes
        self.modes = modes
        
        # Initialize the transforms
        self.transform_image = transform_image
        self.transform_depth = transform_depth
        self.transform_normal = transform_normal

    
    def __len__(self) -> int:
        """Return the size of the dataset

        Returns:
            int: Number of samples
        """
        # Count the number of files in the image directory
        # return len(os.listdir(self.images_directory))
        return len(self.traversal_costs_frame)

    
    def __getitem__(self, idx: int) -> tuple:
        """Allow to access a sample by its index

        Args:
            idx (int): Index of a sample

        Returns:
            tuple: Sample at index idx
            ([multimodal_image,
              traversal_cost,
              traversability_label,
              linear_velocity])
        """
        # Get the image name at index idx
        image_name = os.path.join(
            self.images_directory,
            self.traversal_costs_frame.loc[idx, "image_id"])
        
        # Set an empty list to store the images to concatenate
        multimodal_image = []
        
        if "c" in self.modes:
        
            # Read the RGB image
            image = Image.open(image_name + ".png")

            # Eventually apply transforms to the image
            if self.transform_image:
                image = self.transform_image(image)
                
            # Append the image to the list
            multimodal_image.append(image)
        
        if "d" in self.modes:
            
            # Read the depth image
            # depth_image = tifffile.imread(image_name + "d.tiff")
            depth_image = Image.open(image_name + "d.png")

            # Eventually apply transforms to the depth image
            if self.transform_depth:
                depth_image = self.transform_depth(depth_image)
            
            # Append the depth image to the list
            multimodal_image.append(depth_image)
        
        if "n" in self.modes:
            
            # Read the normal map
            # normal_map = tifffile.imread(image_name + "n.tiff")
            normal_map = Image.open(image_name + "n.png")

            # Eventually apply transforms to the normal map
            if self.transform_normal:
                normal_map = self.transform_normal(normal_map)
                
            # Append the normal map to the list
            multimodal_image.append(normal_map)
        
        # Concatenate the images
        multimodal_image = torch.cat(multimodal_image).float()
        
        # Get the corresponding traversal cost
        traversal_cost =\
            self.traversal_costs_frame.loc[idx,
                                           "traversal_cost"].astype(np.float32)
        
        # Get the corresponding traversability label
        traversability_label =\
            self.traversal_costs_frame.loc[idx, "traversability_label"]
        
        # Get the corresponding linear velocity
        linear_velocity = self.traversal_costs_frame.loc[idx,
                                                         "linear_velocity"]

        return multimodal_image,\
               traversal_cost,\
               traversability_label,\
               linear_velocity
