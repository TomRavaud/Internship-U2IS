#!/bin/bash

# Set some variables

# The date and time will be appended to this prefix
FILE_PREFIX="/media/ssd/forest"

# The maximum duration of a bag file in seconds before splitting it
BAG_MAX_DURATION=180  # [s]

# Size of the internal buffer (if this value is too small,
# messages can be dropped)
BUFFER_SIZE=2048  # [MB, default: 256]

# Set the list of topics to be recorded
TOPICS="/zed_node/depth/depth_registered
        /odometry/filtered/local
        /zed_node/rgb/image_rect_color
        /imu/data"


# Prompt the user for inputs
read -p "Enter a file prefix (default: $FILE_PREFIX): " user_prefix
read -p "Enter a maximum duration (in s) before splitting (default $BAG_MAX_DURATION): " user_duration

# Check if the user inputs are not empty
if [ -n "$user_prefix" ]; then
	# Modify the default value if the user input is not empty
	FILE_PREFIX="$user_prefix"
fi

if [ -n "$user_duration" ]; then
	BAG_MAX_DURATION="$user_duration"
fi

# Add a trap to catch SIGINT in order to stop the node cleanly
#trap "rosnode kill /bagger" SIGINT

# Record the topics' messages
rosbag record -o "$FILE_PREFIX" -b "$BUFFER_SIZE" --split --duration="$BAG_MAX_DURATION" $TOPICS __name:=bagger
