#####################################################
## Parameters for the visualization of the results ##
#####################################################

# Set a color for each terrain class
colors = {
    "road_easy": "black",
    "road_medium": "grey",
    "forest_dirt_easy": "brown",
    "dust": "orange",
    "forest_leaves": "red",
    "forest_dirt_medium": "darkgoldenrod",
    "gravel_easy": "cyan",
    "grass_easy": "lime",
    "grass_medium": "limegreen",
    "gravel_medium": "blue",
    "forest_leaves_branches": "pink",
    "forest_dirt_stones_branches": "purple",
    }


##########################################
## Discretization of the traversal cost ##
##########################################

# Number of bins to digitized the traversal cost
NB_BINS = 10

# Set the binning strategy
BINNING_STRATEGY = "kmeans"
