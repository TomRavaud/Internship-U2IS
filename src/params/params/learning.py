import torch


#########################
## Learning parameters ##
#########################

# Define the data to be used
DATASET = "../../../datasets/dataset_multimodal_siamese/"

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

LEARNING = {"batch_size": 32,
            "nb_epochs": 50,
            "learning_rate": 1e-4,
            "weight_decay": 0.001,
            "momentum": 0.9}


####################################
## Images' transforms parameters ##
####################################

IMAGE_SHAPE = (70, 210)

JITTER_PARAMS = {"brightness": 0.5,
                 "contrast": 0.5}

NORMALIZE_PARAMS = {"rbg": {"mean": torch.tensor([0.4769, 0.5044, 0.4665]),
                            "std": torch.tensor([0.1944, 0.1849, 0.1926])},
                    "depth": {"mean": torch.tensor([0.8172]),
                              "std": torch.tensor([1.1132])},
                    "normal": {"mean": torch.tensor([-2.629e-4, 0.4064, 0.6351]),
                               "std": torch.tensor([0.3747, 0.4044, 0.2807])}
                    }


####################
## Network design ##
####################

# Set the parameters for the network
NET_PARAMS = {"nb_input_channels": 3,
              "nb_input_features": 1,
              "nb_classes": 10}
