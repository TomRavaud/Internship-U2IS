import rosbag
import rospy

import numpy as np



# IMU topic name
IMU_TOPIC = "imu/data"

# Odometry topic name
ODOM_TOPIC = "/odometry/filtered"

IMU_SAMPLE_RATE = 43


# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_grass_wood.bag"

bag = rosbag.Bag(FILE)
_, _, t_start = next(iter(bag.read_messages(topics=[ODOM_TOPIC])))

"""
tom_path_grass:
- grass: 175 - 190
- grass to path: 87 - 95
- path: 102 - 108

tom_road:
- road: 30 - 50

tom_grass_wood:
- sidewalk: 7 - 12 (0.5)
- paved way: 17 - 20
- grass mud: 90 - 100
- grass - leaves: 137 - 147 (0.5)
- leaves: 166 - 176
"""
duration_start = 166
duration_end = 176


z_acceleration = []
roll_rate = []
pitch_rate = []

for _, msg_imu, _ in bag.read_messages(
    topics=[IMU_TOPIC],
    start_time=t_start+rospy.Duration(duration_start),
    end_time=t_start+rospy.Duration(duration_end)):
    
    z_acceleration.append(msg_imu.linear_acceleration.z)
    roll_rate.append(msg_imu.angular_velocity.x)
    pitch_rate.append(msg_imu.angular_velocity.y)

nb_samples = len(z_acceleration)

data_to_save = np.zeros((nb_samples, 4))

x_velocity = []

for _, msg_odom, _ in bag.read_messages(
    topics=[ODOM_TOPIC],
    start_time=t_start+rospy.Duration(duration_start),
    end_time=t_start+rospy.Duration(duration_end)):
    
    x_velocity.append(msg_odom.twist.twist.linear.x)


data_to_save[:, 0] = z_acceleration
data_to_save[:, 1] = roll_rate
data_to_save[:, 2] = pitch_rate
data_to_save[:, 3] = x_velocity[:nb_samples]

np.save("leaves_t10_v1.npy", data_to_save)
