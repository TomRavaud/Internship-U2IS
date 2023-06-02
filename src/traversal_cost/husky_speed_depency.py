"""
A ROS node that makes the Husky robot move back and forth at different speeds.
The goal is to collect vibration data (roll rate, pitch rate and vertical
acceleration data from IMU measurements) at different speeds and on different
terrains to design a traversal cost function.
"""
#!/usr/bin/env python3

# ROS - Python librairies
import rospy
from geometry_msgs.msg import Twist

# Python libraries
import numpy as np


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("back_and_forth")

    # Initialize a publisher to the robot velocity command topic
    pub_velocity = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    
    # Rate at which velocity commands are published on the topic
    PUBLISHING_RATE = 50
    rate = rospy.Rate(PUBLISHING_RATE)
    
    # Distance the robot will travel to collect data
    DISTANCE = 5  # meters
    
    # Create the range of speeds to go through
    min_velocity = 0.2
    max_velocity = 1.
    nb_velocities = 5
    positive_velocities = np.linspace(min_velocity, max_velocity, nb_velocities)
    velocities = np.ravel([positive_velocities, -positive_velocities], order="F")
    print(velocities)
    
    # Instantiate a Twist object to store the velocity command
    velocity_command = Twist()
    
    for velocity in velocities:
        
        distance_travelled = 0
        velocity_command.linear.x = velocity
        
        while not rospy.is_shutdown() and distance_travelled < DISTANCE:
            pub_velocity.publish(velocity_command)
            distance_travelled += np.abs(velocity)*1/PUBLISHING_RATE
            rate.sleep()
        
        rospy.Rate(0.3).sleep()
