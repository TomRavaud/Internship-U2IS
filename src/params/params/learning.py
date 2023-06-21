import torch


#########################
## Learning parameters ##
#########################

# Define the data to be used
DATASET = "../../../datasets/dataset_multimodal_siamese_png/"

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Set learning parameters
LEARNING = {"batch_size": 32,
            "nb_epochs": 40,
            "learning_rate": 1e-4,
            "weight_decay": 0.001,
            "momentum": 0.9}


####################################
## Images' transforms parameters ##
####################################

IMAGE_SHAPE = (70, 210)

JITTER_PARAMS = {"brightness": 0.5,
                 "contrast": 0.5}

NORMALIZE_PARAMS = {"rbg": {"mean": torch.tensor([0.4710, 0.5030, 0.4580]),
                            "std": torch.tensor([0.1965, 0.1859, 0.1955])},
                    "depth": {"mean": torch.tensor([0.0855]),
                              "std": torch.tensor([0.0684])},
                    "normal": {"mean": torch.tensor([0.4981, 0.5832, 0.8387]),
                               "std": torch.tensor([0.1720, 0.1991, 0.1468])}
                    }


####################
## Network design ##
####################

# Set the parameters for the network
NET_PARAMS = {"nb_input_channels": 7,
              "nb_input_features": 1,
              "nb_classes": 10}


#######################################
## Saving the weights of the network ##
#######################################

# The name of the file in which the weights of the network will be saved
PARAMS_FILE = "network.params"


############################
## Output of the training ##
############################

# The name of the directory in which the logs will be saved
LOG_DIR = None
