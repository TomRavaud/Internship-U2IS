"""
Script to build the dataset for the Siamese network. A dataset is a folder
with the following structure:

dataset_{name}
├── features
│   ├── 000.npy
│   ├── 001.npy
│   └── ...
├── labels.csv
├── pairs.csv
├── pairs_test.csv
└── pairs_train.csv

where:
- xxx.npy is a numpy array containing features extracted from the signal
xxx
- labels.csv is a csv file containing the labels of the signals (terrain
class and linear velocity of the robot)
- pairs.csv is a csv file containing the pairs of signals to compare (the
signal with id1 is supposed to be extracted from a terrain with a lower
traversal cost than the associated signal with id2)
- pairs_train.csv contains the training set of pairs
- pairs_test.csv contains the test set of pairs
"""


# Python packages
import pandas as pd
import numpy as np
import itertools  # To get combinations of elements in a list
import os
import sys
import rosbag
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import csv
import shutil
from sklearn.model_selection import train_test_split
from matplotlib.widgets import SpanSelector  # To select a region of the plot

# Custom modules and packages
import params.siamese
import params.robot
import traversalcost.utils


def print_message():
    """
    Print a message to explain how to label the data
    """
    print("1. Click and drag to select a region of the plot")
    print("2. Enter the terrain class")
    print("3. Enter the linear velocity of the robot")
    print("4. Enter the number of sub-regions to divide the selected region into\n")
    print("Close the plot window to go to the next bag file, or to exit\n")
    

class SiameseDatasetBuilder():
    """
    Class to build the dataset for the Siamese network
    """
    def __init__(self, name):
        """Constructor of the class

        Args:
            name (string): The name to give to the dataset
        """
        # Name of the dataset
        self.name = name
        
        # Get the absolute path of the current directory
        directory = os.path.abspath(os.getcwd())

        # Set the name of the directory which will store the dataset
        self.dataset_directory = directory +\
                                 "/src/traversal_cost/datasets/dataset_" +\
                                 self.name
        
        try:  # A new directory is created if it does not exist yet
            os.mkdir(self.dataset_directory)
            print(self.dataset_directory + " folder created\n")

        except OSError:  # Display a message if it already exists and quit
            print("Existing directory " + self.dataset_directory)
            print("Aborting to avoid overwriting data\n")
            sys.exit(1)  # Stop the execution of the script
        
        # Create a sub-directory to store signals features
        self.features_directory = self.dataset_directory + "/features"

        # Create a directory if it does not exist yet
        try:
            os.mkdir(self.features_directory)
            print(self.features_directory + " folder created\n")
        except OSError:
            pass
        
        # Create a csv file to store the pairs of signals features
        self.csv_pairs = self.dataset_directory + "/pairs.csv"
        
        # Create a csv file to store labels of signals
        self.csv_labels = self.dataset_directory + "/labels.csv"
    
    
    def manual_labeling(self, files):
        """Label the dataset manually

        Args:
            files (list): List of bag files
        """
        # Initialize the index of the example to label
        self.example_index = 0
        
        # Open the csv file to contain the labels in write mode
        file_labels = open(self.csv_labels, "w")
        
        # Create a csv writer
        file_labels_writer = csv.writer(file_labels, delimiter=",")
        
        # Write the first row (columns title)
        headers = ["id",
                   "terrain_class",
                   "linear_velocity",
                   "file",
                   "start_index",
                   "end_index"]
        file_labels_writer.writerow(headers)
        
        # Go through the bagfiles
        for file in files:
            
            print("Reading file: " + file + "\n")
            
            # Open the bag file
            bag = rosbag.Bag(file)
            
            # Define lists to store IMU signals
            roll_velocity_values = []
            pitch_velocity_values = []
            vertical_acceleration_values = []

            # Go through the IMU topic of the bag file and store the signals
            for _, msg, t in bag.read_messages(topics=[params.robot.IMU_TOPIC]):
                roll_velocity_values.append(msg.angular_velocity.x)
                pitch_velocity_values.append(msg.angular_velocity.y)
                vertical_acceleration_values.append(msg.linear_acceleration.z - 9.81)

            # Open a figure and plot the roll velocity signal
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(roll_velocity_values, "b")
            
            ax.set_title(f"Roll velocity signal\n({file.split('/')[-1]})")
            ax.set_xlabel("Measurement index")
            ax.set_ylabel("Roll velocity [rad/s]")


            def onselect(xmin, xmax):
                """Callback function called each time a region of the plot is selected

                Args:
                    xmin (float): Lower bound of the selected region
                    xmax (float): Upper bound of the selected region
                """
                # Use the global keyword to refer to the variable example_index defined
                # outside of the function
                # global example_index

                # Ask the user to enter the terrain class and the linear velocity of
                # the robot
                terrain_class = input(
                    f"Enter the terrain class {params.siamese.ORDERED_TERRAIN_CLASSES}: "
                )
                linear_velocity = float(input(
                    f"Enter the linear velocity {params.siamese.LINEAR_VELOCITIES}[m/s]: "
                ))

                # Ask the user to enter the number of sub-regions to divide the
                # selected region into
                nb_subregions = int(input("Enter the number of sub-regions: "))

                # Divide the selected region into sub-regions
                x = np.linspace(xmin, xmax, nb_subregions + 1)

                for i in range(nb_subregions):

                    # Extract features from the signals
                    features = traversalcost.utils.get_features(
                        roll_velocity_values[int(x[i]):int(x[i+1])],
                        pitch_velocity_values[int(x[i]):int(x[i+1])],
                        vertical_acceleration_values[int(x[i]):int(x[i+1])],
                        params.siamese.FEATURES)
                    
                    # Give the example a name
                    example_name = f"{self.example_index:03d}"

                    # Save the features in a numpy file
                    np.save(self.dataset_directory + "/features/" + example_name + ".npy",
                            features)

                    # Write the terrain class and the linear velocity of the robot in the
                    # csv file
                    file_labels_writer.writerow([example_name,
                                                 terrain_class,
                                                 linear_velocity,
                                                 file,
                                                 int(x[i]),
                                                 int(x[i+1])])

                    # Increment the index of the example to label
                    self.example_index += 1


                # Clear the console
                os.system("clear")

                # Print a message to the user
                print_message()


            # Print a message to the user
            print_message()

            # Create a span selector object that gets a selected region from a plot and
            # runs a callback function
            span = SpanSelector(ax,
                                onselect,
                                direction="horizontal",
                                useblit=True,
                                span_stays=True,
                                rectprops=dict(alpha=0.5, facecolor='red'))

            plt.show()

        # Close the csv file
        file_labels.close()
    
    
    def labeling_from_file(self, csv_labels, files):
        """Read the labels from an existing csv file

        Args:
            files (list): List of bag files
        """
        # Open the csv file which contains the labels in read mode
        file_labels = pd.read_csv(csv_labels,
                                  converters={"id": str})
        
        # Go through the bagfiles
        for file in files:
            
            print("Reading file: " + file + "\n")
            
            # Get the labels of the current file
            labels = file_labels[file_labels["file"] == file]
            
            # Open the bag file
            bag = rosbag.Bag(file)
            
            # Define lists to store IMU signals
            roll_velocity_values = []
            pitch_velocity_values = []
            vertical_acceleration_values = []

            # Go through the IMU topic of the bag file and store the signals
            for _, msg, t in bag.read_messages(topics=[params.robot.IMU_TOPIC]):
                roll_velocity_values.append(msg.angular_velocity.x)
                pitch_velocity_values.append(msg.angular_velocity.y)
                vertical_acceleration_values.append(msg.linear_acceleration.z - 9.81)
            
            # Go through the labels
            for i in range(len(labels.index)):
                
                # Get the id of the signal and the start and end indices of the
                # sub-signals
                id, _, _, _, xmin, xmax = labels.iloc[i]
                
                # Extract features from the signals
                features = traversalcost.utils.get_features(
                    roll_velocity_values[xmin:xmax],
                    pitch_velocity_values[xmin:xmax],
                    vertical_acceleration_values[xmin:xmax],
                    params.siamese.FEATURES)
                    
                # Save the features in a numpy file
                np.save(self.dataset_directory + "/features/" +
                        id + ".npy",
                        features)

                # Copy the csv file in the dataset directory
                shutil.copy(csv_labels, self.dataset_directory + "/labels.csv")


    def find_and_write_pairs(self):
        """
        Find pairs of signals that can be compared based on basic assumptions
        (same terrain class or same linear velocity), and write them in a csv
        file
        """
        # Open the csv file containing the labels
        labels = pd.read_csv(self.csv_labels, converters={"id": str})
        
        # Get all the possible pairs of elements
        combinations = np.array(list(itertools.combinations(labels["id"], 2)))
        
        # Get indexes the pairs in which the two elements are of different classes AND
        # different velocities OR the two elements are of the same class AND same
        # velocity
        pairs_to_remove = []

        for i in range(len(combinations)):

            index1, index2 = combinations[i]
            
            if (labels.loc[int(index1), "terrain_class"] !=\
                labels.loc[int(index2), "terrain_class"] and\
                labels.loc[int(index1), "linear_velocity"] !=\
                labels.loc[int(index2), "linear_velocity"]) or\
               (labels.loc[int(index1), "terrain_class"] ==\
                labels.loc[int(index2), "terrain_class"] and\
                labels.loc[int(index1), "linear_velocity"] ==\
                labels.loc[int(index2), "linear_velocity"]) or\
               (labels.loc[int(index1), "terrain_class"] ==\
                labels.loc[int(index2), "terrain_class"] and\
                np.abs(labels.loc[int(index1), "linear_velocity"] -\
                labels.loc[int(index2), "linear_velocity"]) > 0.2 + 1e-4):

                pairs_to_remove.append(i)

            # If the pair is not to be removed, check that the first element must have
            # a lower cost than the second one, else swap the indexes
            else:
                if params.siamese.ORDERED_TERRAIN_CLASSES.index(
                    labels.loc[int(index1), "terrain_class"]) >\
                   params.siamese.ORDERED_TERRAIN_CLASSES.index(
                    labels.loc[int(index2), "terrain_class"]) or\
                   labels.loc[int(index1), "linear_velocity"] >\
                    labels.loc[int(index2), "linear_velocity"]:

                    # Swap the indexes
                    combinations[i] = [index2, index1]

        # Remove incorrect pairs from the array of combinations
        combinations = np.delete(combinations, pairs_to_remove, axis=0)
        
        # Create a dataframe with the pairs
        dataframe = pd.DataFrame(combinations, columns=["id1", "id2"])
        
        # Store the pairs in a csv file
        dataframe.to_csv(self.csv_pairs, index=False)
        
    
    def create_train_test_splits(self):
        """
        Split the dataset into training and testing sets
        """
        # Read the CSV file into a Pandas dataframe (read id values as
        # strings to keep leading zeros)
        dataframe = pd.read_csv(self.csv_pairs, converters={"id1": str,
                                                            "id2": str})
        
        # Split the dataset randomly into training and testing sets
        dataframe_train, dataframe_test =\
            train_test_split(dataframe,
                             train_size=params.siamese.TRAIN_SIZE +
                                        params.siamese.VAL_SIZE)

        # Store the train and test splits in csv files
        dataframe_train.to_csv(self.dataset_directory + "/pairs_train.csv", index=False)
        dataframe_test.to_csv(self.dataset_directory + "/pairs_test.csv", index=False)
    
    
    def generate_features_description(self):
        """
        Generate a text file that describes how the features are extracted
        from the signals
        """
        # Open the text file
        description_file = open(self.dataset_directory +
                                "/features_description.txt", "w")
        
        # Generate the table which contains the description of the features
        table = traversalcost.utils.generate_description(
            params.siamese.FEATURES)
        
        # Write the table in the text file
        description_file.write(table)
        
        # Close the text file
        description_file.close()


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    dataset = SiameseDatasetBuilder(name="40Hz_dwt_hard")
    
    # List of the bag files to be processed
    files=[
        "bagfiles/raw_bagfiles/speed_dependency/grass.bag",
        "bagfiles/raw_bagfiles/speed_dependency/road.bag",
        "bagfiles/raw_bagfiles/speed_dependency/sand.bag",
    ]
    
    # Choose between manual labeling or labeling from a csv file
    # dataset.manual_labeling(files=files)
    
    # If you choose the labeling from a csv file, you must provide the csv
    # file which contains the labels. It allows us to extract new features
    # from already labeled signals
    dataset.labeling_from_file(
        csv_labels="src/traversal_cost/datasets/dataset_40Hz_variance/labels.csv",
        files=files)
    
    dataset.generate_features_description()
    
    dataset.find_and_write_pairs()
    
    dataset.create_train_test_splits()
