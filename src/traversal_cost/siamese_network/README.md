# Siamese Network for traversal cost estimation

This folder contains the code used to train a network to predict a terrain traversal cost from IMU signals recorded by the robot. In particular, only three signals are considered:

* the robot's roll rate
* the robot's pitch rate
* the robot's vertical acceleration

Features are extracted from these signals and set as inputs of the network. The network is trained thanks to a Siamese structure, based on basic assumptions:

* for a given terrain, the traversal cost is higher at high speed than at low speed
* for a given velocity, the traversal cost is higher on grass than on sand, higher on sand than on road, etc.

This folder contains the following files:

* **create_dataset.py**: a script to create a dataset from bag files containing IMU data
* **dataset.py**: the PyTorch dataset
* **loss.py**: the custom loss function to train the network
* **main.ipynb**: a notebook which gathers all the training steps, from data loading to testing
* **model.py**: the model used to predict the traversal cost
* **test.py**: the test function
* **train.py**: the train function (one epoch)
* **validate.py**: the validate function (one epoch)
