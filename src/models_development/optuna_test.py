import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch import optim

import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "figure.figsize": (10,5),
})

import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

from tqdm import tqdm


# Define the data to be used
DATASET = "datasets/dataset_3+8bags_3var3sc_regression_classification_kmeans_split/"


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
        # Count the number of files in the image directory
        # return len(os.listdir(self.images_directory))
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
                                  self.traversal_costs_frame.loc[idx, "image_id"])
        
        # Read the image
        image = Image.open(image_name)
        
        # Eventually apply transforms to the image
        if self.transform:
            image = self.transform(image)
        
        # Get the corresponding traversal cost
        traversal_cost = self.traversal_costs_frame.loc[idx, "traversal_cost"]
        
        # Get the corresponding traversability label
        traversability_label = self.traversal_costs_frame.loc[idx, "traversability_label"]

        return image, traversal_cost, traversability_label
 

def train(model, device, train_loader, optimizer, criterion):
    # Training
    train_loss = 0.
    
    # Configure the model for training
    # (good practice, only necessary if the model operates differently for
    # training and validation)
    model.train()
    
    # Add a progress bar
    train_loader_pbar = tqdm(train_loader, unit="batch")
    
    # Loop over the training batches
    for images, traversal_costs, _ in train_loader_pbar:
        
        # Move images and traversal scores to GPU (if available)
        images = images.to(device)
        traversal_costs = traversal_costs.type(torch.FloatTensor).to(device)
        
        # Zero out gradients before each backpropagation pass, to avoid that
        # they accumulate
        optimizer.zero_grad()
        
        # Perform forward pass
        predicted_traversal_costs = model(images)
        
        # Compute loss 
        loss = criterion(predicted_traversal_costs[:, 0], traversal_costs)
        
        # Print the batch loss next to the progress bar
        train_loader_pbar.set_postfix(batch_loss=loss.item())
        
        # Perform backpropagation (update weights)
        loss.backward()
        
        # Adjust parameters based on gradients
        optimizer.step()
        
        # Accumulate batch loss to average over the epoch
        train_loss += loss.item()
    
    # Compute the loss
    train_loss /= len(train_loader)
    
    return train_loss


def validate(model, device, val_loader, criterion):
    # Validation
    val_loss = 0.
    
    # Configure the model for testing
    # (turn off dropout layers, batchnorm layers, etc)
    model.eval()
    
    # Add a progress bar
    val_loader_pbar = tqdm(val_loader, unit="batch")
    
    # Turn off gradients computation (the backward computational graph is built during
    # the forward pass and weights are updated during the backward pass, here we avoid
    # building the graph)
    with torch.no_grad():
        # Loop over the validation batches
        for images, traversal_costs, _ in val_loader_pbar:

            # Move images and traversal scores to GPU (if available)
            images = images.to(device)
            traversal_costs = traversal_costs.type(torch.FloatTensor).to(device)
            
            # Perform forward pass (only, no backpropagation)
            predicted_traversal_costs = model(images)

            # Compute loss
            loss = criterion(predicted_traversal_costs[:, 0], traversal_costs)

            # Print the batch loss next to the progress bar
            val_loader_pbar.set_postfix(batch_loss=loss.item())

            # Accumulate batch loss to average over the epoch
            val_loss += loss.item()
            
    
    # Compute the loss
    val_loss /= len(val_loader)
    
    return val_loss

def objective2(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

def objective(trial):
    
    print("\nTrial", trial.number)
    
    # Use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Select parameter uniformaly within the range
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # wd = trial.suggest_float("wd", 1e-6, 1e-1, log=True)

    # Compose several transforms together to be applied to training data
    # (Note that transforms are not applied yet)
    train_transform = transforms.Compose([
        # Reduce the size of the images
        # (if size is an int, the smaller edge of the
        # image will be matched to this number and the ration is kept)
        # transforms.Resize(100),
        transforms.Resize((70, 210)),

        # Perform horizontal flip of the image with a probability of 0.5
        transforms.RandomHorizontalFlip(p=0.5),

        # Modify the brightness and the contrast of the image
        transforms.ColorJitter(contrast=0.5, brightness=0.5),

        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),

        # Add some random gaussian noise to the image
        transforms.Lambda(lambda x: x + (0.001**0.5)*torch.randn(x.shape)),

        # Normalize a tensor image with pre-computed mean and standard deviation
        # (based on the data used to train the model(s))
        # (be careful, it only works on torch.*Tensor)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Define a different set of transforms testing
    # (for instance we do not need to flip the image)
    test_transform = transforms.Compose([
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
    
    # Create a Dataset for training
    train_set = TraversabilityDataset(
        traversal_costs_file=DATASET+"traversal_costs_train.csv",
        images_directory=DATASET+"images_train",
        transform=train_transform
    )

    # Create a Dataset for validation
    val_set = TraversabilityDataset(
        traversal_costs_file=DATASET+"traversal_costs_train.csv",
        images_directory=DATASET+"images_train",
        transform=test_transform
    )
    
    # Set the train dataset size
    # 70% of the total data is used for training, 15% for validation
    # and 15% for testing
    train_size = 70/(100-15)

    # Splits train data indices into train and validation data indices
    train_indices, val_indices = train_test_split(range(len(train_set)), train_size=train_size)

    # Extract the corresponding subsets of the train dataset
    train_set = Subset(train_set, train_indices)
    val_set = Subset(val_set, val_indices)

    BATCH_SIZE = 32

    # Combine a dataset and a sampler, and provide an iterable over the dataset
    # (setting shuffle argument to True calls a RandomSampler, and avoids to
    # have to create a Sampler object)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,  # Asynchronous data loading and augmentation
        pin_memory=True,  # Increase the transferring speed of the data to the GPU
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    
    # Load the pre-trained ResNet model
    model = models.resnet18().to(device=device)
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device=device)
    # Replace the last layer by a fully-connected one with 1 output
    model.fc = nn.Linear(model.fc.in_features, 1, device=device)
    # Initialize the last layer using Xavier initialization
    # nn.init.xavier_uniform_(model.fc.weight)
    
    # Define the loss function
    criterion = nn.MSELoss()

    # Get all the parameters excepts the weights and bias of fc layer
    base_params = [param for name, param in model.named_parameters()
                   if name not in ["fc.weight", "fc.bias"]]

    # Define the optimizer, with a greater learning rate for the new fc layer
    optimizer = optim.SGD([
        {"params": base_params},
        {"params": model.fc.parameters(), "lr": lr},
    ],
        lr=lr, momentum=0.9, weight_decay=0.01)
    # optimizer = optim.SGD([
    #     {"params": base_params},
    #     {"params": model.fc.parameters(), "lr": 1e-3},
    # ],
    #     lr=1e-4, momentum=0.9, weight_decay=wd)

    # An epoch is one complete pass of the training dataset through the network
    NB_EPOCHS = 100
    patience = 5
    
    # Train the model for multiple epochs
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(NB_EPOCHS):
        
        train(model, device, train_loader, optimizer, criterion)
        val_loss = validate(model, device, val_loader, criterion)
        
        # Communicate with Optuna about the progress of the trial
        # trial.report(val_loss, epoch)

        # Early stopping based on validation loss: stop the training if the
        # loss has not improved for the last 5 epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        elif epoch - best_epoch >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return best_val_loss



# Samplers: GridSampler, RandomSampler, TPESampler, CmaEsSampler, PartialFixedSampler, NSGAIISampler, QMCSampler
# Pruners: MedianPruner, NopPruner, PatientPruner, PercentilePruner, SuccessiveHalvingPruner, HyperbandPruner, ThresholdPruner
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    # pruner=optuna.pruners.MedianPruner()
    )
study.optimize(objective, n_trials=10)

best_params = study.best_params
found_learning_rate = best_params["lr"]
# found_weight_decay = best_params["wd"]
print("Found learning rate: ", found_learning_rate)
# print("Found weight decay: ", found_weight_decay)


# Plot functions
axes_optimization_history = plot_optimization_history(study)
axes_optimization_history.loglog()
axes_optimization_history.legend()

# # When pruning
# axes_intermediate_values = plot_intermediate_values(study)

plot_parallel_coordinate(study)

# # Should be at least 2 parameters to see something
# plot_contour(study)

plot_param_importances(study)

plt.show()
