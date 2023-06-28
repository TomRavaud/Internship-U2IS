"""
Script to build the dataset for the network. A dataset is a folder
with the following structure:

dataset_{name}
├── features
│   ├── 000.npy
│   ├── 001.npy
│   └── ...
├── features_description.txt
├── labels.csv
├── traversal_costs.csv
├── traversal_costs_test.csv
└── traversal_costs_train.csv

where:
- xxx.npy is a numpy array containing features extracted from the signal
xxx
- features_description.txt is a text file containing the description of the
features extracted from the signals
- labels.csv is a csv file containing the labels of the signals (terrain
class and linear velocity of the robot)
- traversal_costs.csv is a csv file containing the traversal cost of the
signals
- traversal_costs_train.csv contains the training set of traversal costs
- traversal_costs_test.csv contains the test set of traversal costs
"""

# Python packages
import pandas as pd
import numpy as np
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
import params.supervised_learning
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
    

class SupervisedDatasetBuilder():
    """
    Class to build the dataset for the Supervised Learning network
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
        
        # Create a list to store bag files paths
        bag_files = []
        
        # Go through the list of files
        for file in files:
            
            # Check if the file is a bag file
            if os.path.isfile(file) and file.endswith(".bag"):
                bag_files.append(file)
            
            # If the path links to a directory, go through the files inside it
            elif os.path.isdir(file):
                bag_files.extend(
                    [file + f for f in os.listdir(file) if f.endswith(".bag")])
        
        # Go through the bagfiles
        for file in bag_files:
            
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
            
            ax.set_title(f"Roll velocity signal\n({(file.split('/')[-1]).replace('_', '-')})")
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

                # Ask the user to enter the terrain class and the linear
                # velocity of the robot
                terrain_class = input(
                    f"Enter the terrain class : "
                )
                linear_velocity = float(input(
                    f"Enter the linear velocity [m/s]: "
                ))

                # Ask the user to enter the number of sub-regions to divide the
                # selected region into
                nb_subregions = int(input("Enter the number of sub-regions: "))

                # Divide the selected region into sub-regions
                x = np.linspace(xmin, xmax, nb_subregions + 1)

                for i in range(nb_subregions):
                    
                    print(len(roll_velocity_values[int(x[i]):int(x[i+1])]))
                    
                    # Extract features from the signals
                    features = traversalcost.utils.get_features(
                        roll_velocity_values[int(x[i]):int(x[i+1])],
                        pitch_velocity_values[int(x[i]):int(x[i+1])],
                        vertical_acceleration_values[int(x[i]):int(x[i+1])],
                        params.supervised_learning.FEATURES)
                    
                    # Give the example a name
                    example_name = f"{self.example_index:03d}"

                    # Save the features in a numpy file
                    np.save(self.dataset_directory +
                            "/features/" +
                            example_name + ".npy",
                            features)

                    # Write the terrain class and the linear velocity of the
                    # robot in the csv file
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

            # Create a span selector object that gets a selected region from
            # a plot and runs a callback function
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
                
        # Create a list of store bag files paths
        bag_files = []
        
        # Go through the bagfiles
        for file in files :
            
            # Check if the file is a bag file
            if os.path.isfile(file) and file.endswith(".bag"):
                bag_files.append(file)

            # If the path links to a directory, go through the files inside it
            elif os.path.isdir(file):
                bag_files.extend(
                    [file + f for f in os.listdir(file) if f.endswith(".bag")]
                )
        
        # Go through the bagfiles
        for file in bag_files:
                        
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
                    params.supervised_learning.FEATURES)
                    
                # Save the features in a numpy file
                np.save(self.dataset_directory + "/features/" +
                        id + ".npy",
                        features)

        # Copy the csv file in the dataset directory
        shutil.copy(csv_labels, self.dataset_directory + "/labels.csv")


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
            params.supervised_learning.FEATURES)
        
        # Write the table in the text file
        description_file.write(table)
        
        # Close the text file
        description_file.close()


    def generate_costs(self):    # Same as cost_to_data but much faster
        
        # Import dataframe
        df = pd.read_csv(self.csv_labels,
                         converters={"id": str},
                         usecols = ['id', 'linear_velocity','terrain_class'])
        
        # Import functions and keys from a dictionary 
        terrain_cost_functions = params.supervised_learning.terrain_cost
        
        # Get the set of all different terrains
        labels_unique = set(list(df["terrain_class"]))
        
        # Loop over terrains
        for label in labels_unique:
            
            # Get the cost function related to the current terrain
            cost_function = terrain_cost_functions.get(label)
            
            # Apply the cost function to the linear velocity (and store the
            # result in a new column)
            df.loc[df["terrain_class"] == label, "cost"] = cost_function(df["linear_velocity"])
        
        
        # Save the dataframe in a csv file
        # df[["id", "cost"]].to_csv(self.dataset_directory +
        #                           "/traversal_costs.csv",
        #                           index=False)
        df.to_csv(self.dataset_directory +
                  "/traversal_costs.csv",
                  index=False)


    def create_train_test_splits(self):
        """
        Split the dataset into training and testing sets
        """
        # Read the CSV file into a Pandas dataframe (read id values as
        # strings to keep leading zeros)
        dataframe = pd.read_csv(self.dataset_directory + "/traversal_costs.csv",
                                converters={"id": str})

        dataframe_train, dataframe_test =\
            train_test_split(dataframe,
                             train_size=params.supervised_learning.TRAIN_SIZE +
                             params.supervised_learning.VAL_SIZE)

        # Store the train and test splits in csv files
        dataframe_train.to_csv(self.dataset_directory +
                               "/traversal_costs_train.csv",
                               index=False)
        dataframe_test.to_csv(self.dataset_directory +
                              "/traversal_costs_test.csv",
                              index=False)


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    dataset = SupervisedDatasetBuilder(name="train_test")
    
    # List of the bag files to be processed
    files=[
        "bagfiles/raw_bagfiles/Terrains_Samples/"
        ]
    
    # Choose between manual labeling or labeling from a csv file
    # dataset.manual_labeling(files=files)
    
    # If you choose the labeling from a csv file, you must provide the csv
    # file which contains the labels. It allows us to extract new features
    # from already labeled signals
    dataset.labeling_from_file(
            csv_labels="src/traversal_cost/datasets/dataset_200Hz_variance/labels.csv",
            files=files)
    
    dataset.generate_features_description()
        
    dataset.generate_costs()
    
    dataset.create_train_test_splits()
