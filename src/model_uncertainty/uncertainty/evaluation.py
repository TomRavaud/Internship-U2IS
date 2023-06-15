import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Import custom packages
import params.learning


def uncertainty_relevance(model: torch.nn.Module,
                          device: str,
                          criterion_classification: torch.nn.Module,
                          criterion_regression: torch.nn.Module,
                          bins_midpoints: np.ndarray,
                          uncertainty_function: callable,
                          test_function: callable,
                          test_set: torch.utils.data.Dataset,
                          test_regression_loss: float,
                          test_regression_losses: list,
                          uncertainties: list) -> tuple:
    """Compute the regression losses after successive removal of the samples
    with the highest loss and uncertainty

    Args:
        model (torch.nn.Module): The model to test
        device (str): The device to use for the computations
        criterion_classification (torch.nn.Module): The loss function for the
        classification task
        criterion_regression (torch.nn.Module): The loss function for the
        regression task
        bins_midpoints (np.ndarray): The midpoints of the bins used for the
        discretization of the traversal cost
        uncertainty_function (callable): The function to use to compute the
        uncertainty
        test_function (callable): The function to use to compute the test
        test_set (torch.utils.data.Dataset): The test set
        test_regression_loss (float): The regression loss on the test set
        test_regression_losses (list): The list of the regression losses for
        each sample
        uncertainties (list): The list of the uncertainties for each sample

    Returns:
        (list, list): The list of the regression losses after successive
        removal of the samples with the highest loss, the list of the
        regression losses after successive removal of the samples with the
        highest uncertainty
    """
    # Get the number of samples in the test set
    nb_samples = len(test_set)
    
    # Create lists to store the test losses
    test_losses_loss = [test_regression_loss]
    test_losses_uncertainty = [test_regression_loss]
    
    # Concatenate the regression losses and convert to numpy array
    test_regression_losses = torch.cat(test_regression_losses, dim=0).numpy()

    # Concatenate the uncertainties and convert to numpy array
    uncertainties = torch.cat(uncertainties, dim=0).numpy()

    # Loop over the different percentages of samples to keep
    for i in range(1, 10):

        # Calculate the number of samples to keep
        nb_samples_to_keep = int((1 - i*0.1)*nb_samples)
        # Calculate the indices of the samples with the lowest losses
        indices_to_keep_loss =\
            test_regression_losses.argsort()[:nb_samples_to_keep]
        # Calculate the indices of the samples with the lowest uncertainties
        indices_to_keep_uncertainty =\
            uncertainties.argsort()[:nb_samples_to_keep]
        
        # if i == 1:
        #     test_set_display = Subset(
        #         test_set,
        #         indices=uncertainties.argsort()[nb_samples_to_keep:])
            
        #     for tensor, tcost, tclass, vel in test_set_display:
        #         # De-normalize the normalized tensor
        #         tensor_denormalized = transforms.Compose([
        #             transforms.Normalize(
        #                 mean=[0., 0., 0., 0.],
        #                 std=1/std
        #                 ),
        #             transforms.Normalize(
        #                 mean=-mean,
        #                 std=[1., 1., 1., 1.]
        #                 ),
        #             ])(tensor)

        #         # Convert the tensor to a PIL Image
        #         image_denormalized =\
        #             transforms.ToPILImage()(tensor_denormalized)
                
        #         plt.imshow(image_denormalized)
        #         plt.title("De-normalized image")
                
        # Create a new test dataset without the samples with the largest losses
        test_set_loss = Subset(test_set, indices=indices_to_keep_loss)
        # Create a new test dataset without the samples with the largest
        # uncertainties
        test_set_uncertainty = Subset(test_set,
                                      indices=indices_to_keep_uncertainty)

        # Create a new test dataloader without the samples with the largest
        # losses
        test_loader_loss = DataLoader(
            test_set_loss,
            batch_size=params.learning.LEARNING["batch_size"],
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            )
        # Create a new test dataloader without the samples with the largest
        # uncertainties
        test_loader_uncertainty = DataLoader(
            test_set_uncertainty,
            batch_size=params.learning.LEARNING["batch_size"],
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            )

        # Test the model on the new test dataset without the samples with
        # the largest losses
        test_regression_loss_loss = test_function(model, 
                                                  device,
                                                  test_loader_loss,
                                                  criterion_classification,
                                                  criterion_regression,
                                                  bins_midpoints,
                                                  uncertainty_function)[2]
        # Test the model on the new test dataset without the samples with the
        # largest uncertainties
        test_regression_loss_uncertainty = test_function(
            model,
            device,
            test_loader_uncertainty,
            criterion_classification,
            criterion_regression,
            bins_midpoints,
            uncertainty_function)[2]

        # Append the test loss to the list
        test_losses_loss.append(test_regression_loss_loss)
        test_losses_uncertainty.append(test_regression_loss_uncertainty)
        
    return test_losses_loss, test_losses_uncertainty


def uncertainty_relevance_models(models: list,
                                 device: str,
                                 criterion_classification: torch.nn.Module,
                                 criterion_regression: torch.nn.Module,
                                 bins_midpoints: np.ndarray,
                                 uncertainty_function: callable,
                                 test_function: callable,
                                 test_set: torch.utils.data.Dataset,
                                 test_regression_loss: float,
                                 test_regression_losses: list,
                                 uncertainties: list) -> tuple:
    """Compute the regression losses after successive removal of the samples
    with the highest loss and uncertainty

    Args:
        models (list): The models to test
        device (str): The device to use for the computations
        criterion_classification (torch.nn.Module): The loss function for the
        classification task
        criterion_regression (torch.nn.Module): The loss function for the
        regression task
        bins_midpoints (np.ndarray): The midpoints of the bins used for the
        discretization of the traversal cost
        uncertainty_function (callable): The function to use to compute the
        uncertainty
        test_function (callable): The function to use to test the models
        test_set (torch.utils.data.Dataset): The test set
        test_regression_loss (float): The regression loss on the test set
        test_regression_losses (list): The list of the regression losses for
        each sample
        uncertainties (list): The list of the uncertainties for each sample

    Returns:
        list, list: The list of the regression losses after successive removal
        of the samples with the highest loss, the list of the regression losses
        after successive removal of the samples with the highest uncertainty
    """
    # Get the number of samples
    nb_samples = len(test_set)
    
    # Create lists to store the test losses
    test_losses_loss = [test_regression_loss]
    test_losses_uncertainty = [test_regression_loss]
    
    # Concatenate the regression losses and convert to numpy array
    test_regression_losses = torch.cat(test_regression_losses, dim=0).numpy()

    # Concatenate the uncertainties and convert to numpy array
    uncertainties = torch.cat(uncertainties, dim=0).numpy()

    # Loop over the different percentages of samples to keep
    for i in range(1, 10):

        # Calculate the number of samples to keep
        nb_samples_to_keep = int((1 - i*0.1)*nb_samples)
        # Calculate the indices of the samples with the lowest losses
        indices_to_keep_loss =\
            test_regression_losses.argsort()[:nb_samples_to_keep]
        # Calculate the indices of the samples with the lowest uncertainties
        indices_to_keep_uncertainty =\
            uncertainties.argsort()[:nb_samples_to_keep]

        # Create a new test dataset without the samples with the largest losses
        test_set_loss = Subset(test_set, indices=indices_to_keep_loss)
        # Create a new test dataset without the samples with the largest
        # uncertainties
        test_set_uncertainty = Subset(test_set,
                                      indices=indices_to_keep_uncertainty)

        # Create a new test dataloader without the samples with the largest
        # losses
        test_loader_loss = DataLoader(
            test_set_loss,
            batch_size=params.learning.LEARNING["batch_size"],
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            )
        # Create a new test dataloader without the samples with the largest
        # uncertainties
        test_loader_uncertainty = DataLoader(
            test_set_uncertainty,
            batch_size=params.learning.LEARNING["batch_size"],
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            )

        # Test the model on the new test dataset without the samples with the
        # largest losses
        test_regression_loss_loss = test_function(models, 
                                         device,
                                         test_loader_loss,
                                         criterion_classification,
                                         criterion_regression,
                                         bins_midpoints,
                                         uncertainty_function)[0]
        # Test the model on the new test dataset without the samples with the
        # largest uncertainties
        test_regression_loss_uncertainty = test_function(models,
                                                device,
                                                test_loader_uncertainty,
                                                criterion_classification,
                                                criterion_regression,
                                                bins_midpoints,
                                                uncertainty_function)[0]
        
        # Append the test loss to the list
        test_losses_loss.append(test_regression_loss_loss)
        test_losses_uncertainty.append(test_regression_loss_uncertainty)
        
    return test_losses_loss, test_losses_uncertainty
