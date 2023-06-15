# Python packages
import pandas as pd
import random
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
import params.supervised_learning
import params.robot

from traversalcost import utils
    

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
            pass
            #sys.exit(1)  # Stop the execution of the script
        
        # Create a sub-directory to store signals features
        self.features_directory = self.dataset_directory + "/features"

        # Create a directory if it does not exist yet
        try:
            os.mkdir(self.features_directory)
            print(self.features_directory + " folder created\n")
        except OSError:
            pass
        
        # Create a csv file to store labels of signals
        print(f"self.dataset_directory is {self.dataset_directory}")
        self.csv_labels = self.dataset_directory + "/labels.csv"
        print(f"csv file is saved !")
        #shutil.copy("/home/student/Desktop/Stage ENSTA/Internship-U2IS/src/traversal_cost/data_Arnaud/labels.csv",
        #            self.dataset_directory + "/labels.csv")
        
        #sys.exit(1)
        
    
    
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
                   "start_index",       #je peux virer ca ?
                   "end_index"
                   ]
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
    
    
    def labeling_from_file(self, csv_labels, files):
        """Read the labels from an existing csv file

        Args:
            files (list): List of bag files
        """
        # Open the csv file which contains the labels in read mode
        file_labels = pd.read_csv(csv_labels,
                                  converters={"id": str})
                
        #Create a list of store bag files paths
        bag_files = []
        
        # Go through the bagfiles
        for file in files :
            
            #Check if the file is a bag file
            if os.path.isfile(file) and file.endswith(".bag"):
                bag_files.append(file)

            #If the path links to a directory, go through the files inside it
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
                features = utils.get_features(
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
        table = utils.generate_description(
            params.supervised_learning.FEATURES)
        
        # Write the table in the text file
        description_file.write(table)
        
        # Close the text file
        description_file.close()
        
    def cost_to_data(self): #Slower but usefull if generate_cost doesnt give the intented output
        
        #Importing dataframe
        data = pd.read_csv(self.csv_labels, 
                        converters={"id": str, "terrain_class" : str}, 
                        usecols = ['id', 'linear_velocity','terrain_class'])
        
        #Importing function and keys from dictonnary 
        labels_with_function = params.supervised_learning.terrain_cost
        keys_list = labels_with_function.keys()
        
        terrain = data['terrain_class']
        speed = data['linear_velocity']
        idx = data['id']
        
        list_sample = []
        list_cost = []
        
        #Loop over lenght of the dataframe, could take some time if the dataframe is large
        for i in range(len(data)):

            list_sample.append([idx[i], terrain[i], speed[i]])
        
        for key in keys_list:  #Loop is not optimal for big data sets
                               #but works as fast as generate_cost for smaller ones
            
            for sample in list_sample:
                            
                if sample[1] == key :
                                        
                    cost = labels_with_function[key](float(sample[-1]))
                    list_cost.append(cost)
        
    
        #Dictionary of lists  
        dict = {'id': idx, 'cost': list_cost}  

        #Creating new dataframe
        new_df = pd.DataFrame(dict) 
            
        #Saving the dataframe         
        new_df.to_csv(self.dataset_directory + "/traversal_cost.csv", index=False)
                
    def generate_costs(self):    #Same as cost_to_data but much faster
        
        #Importing dataframe        
        data = pd.read_csv(self.csv_labels,
                           converters={"id": str},
                           usecols = ['id', 'linear_velocity','terrain_class'])
        
        #Importing function and keys from dictonnary 
        terrain_cost_from_params = params.supervised_learning.terrain_cost
        Liste_df = []
        Liste_index = data['id']
        
        #Getting the set of all differents terrains
        labels_unique = set(list(data["terrain_class"]))
        
        #Loop over terrains
        for label in labels_unique:
            
            #Creating new dataframe
            df_label = data[data["terrain_class"] == label]            
            cost_function = terrain_cost_from_params.get(label)
                
            df_label["cost"] = cost_function(df_label["linear_velocity"])
            Liste_df.append(df_label)
        
        #Merging dataframes
        full_df = pd.concat(Liste_df)

        #Adding originals id for dataset afterwards
        full_df['id'] = Liste_index

        #Dropping the columns we don't want 
        full_df = full_df.drop(columns=['linear_velocity','terrain_class'])
        
        #Saving data
        full_df.to_csv(self.dataset_directory + "/traversal_cost.csv", index=False)

    
    def create_train_test_splits(self):
        """
        Split the dataset into training and testing sets
        """
        
        # Read the CSV file into a Pandas dataframe (read id values as
        # strings to keep leading zeros)
        dataframe = pd.read_csv(self.dataset_directory + "/traversal_cost.csv", converters={"id": str})

        # Asks the size of the test sample
        trainsize = params.supervised_learning.LENGHT_TEST_SET
                          
        dataframe_train, dataframe_test =\
            train_test_split(dataframe,
                             train_size=1 - trainsize)

        # Store the train and test splits in csv files
        dataframe_train.to_csv(self.dataset_directory + "/traversalcosts_train.csv", index=False)
        dataframe_test.to_csv(self.dataset_directory + "/traversalcosts_test.csv", index=False)


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    dataset = SupervisedDatasetBuilder(name="big_DS_test")
    
    # List of the bag files to be processed
    files=[
        "bagfiles/raw_bagfiles/Terrains_Samples/"
        ]
    
    # Choose between manual labeling or labeling from a csv file
    #dataset.manual_labeling(files=files)
    
    # If you choose the labeling from a csv file, you must provide the csv
    # file which contains the labels. It allows us to extract new features
    # from already labeled signals
    
    dataset.labeling_from_file(
            csv_labels="src/traversal_cost/data_Arnaud/labels.csv",
            files=files)
    
    dataset.generate_features_description()
        
    dataset.generate_costs()
    
    dataset.create_train_test_splits()