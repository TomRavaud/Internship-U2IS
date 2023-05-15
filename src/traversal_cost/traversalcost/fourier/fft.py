import rosbag
import rospy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rcParams['text.usetex'] = True  # Render Matplotlib text with Tex

from scipy.fft import rfft, rfftfreq


# IMU topic name
IMU_TOPIC = "imu/data"

# Odometry topic name
ODOM_TOPIC = "/odometry/filtered"

IMU_SAMPLE_RATE = 43


fig = plt.figure("3D")
ax = fig.add_subplot(projection='3d')


# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_path_grass.bag"

bag = rosbag.Bag(FILE)
_, _, t_start = next(iter(bag.read_messages(topics=[ODOM_TOPIC])))


z_acceleration = []
roll_rate = []
pitch_rate = []

for _, msg_imu, t_imu in bag.read_messages(topics=[IMU_TOPIC], start_time=t_start+rospy.Duration(175), end_time=t_start+rospy.Duration(190)):
    z_acceleration.append(msg_imu.linear_acceleration.z)
    roll_rate.append(msg_imu.angular_velocity.x)
    pitch_rate.append(msg_imu.angular_velocity.y)


print(len(z_acceleration))

z_acceleration = z_acceleration - np.float32(9.81)

z_acceleration_f = rfft(z_acceleration)
roll_rate_f = rfft(roll_rate)
pitch_rate_f = rfft(pitch_rate)

plt.figure("Fourier grass")
plt.subplot(131)
plt.plot(rfftfreq(len(z_acceleration), 1/IMU_SAMPLE_RATE),
         np.abs(z_acceleration_f))
plt.subplot(132)
plt.plot(rfftfreq(len(roll_rate), 1/IMU_SAMPLE_RATE),
         np.abs(roll_rate_f))
plt.subplot(133)
plt.plot(rfftfreq(len(pitch_rate), 1/IMU_SAMPLE_RATE),
         np.abs(pitch_rate_f))

# x_velocity = []
# for _, msg_odom, t_odom in bag.read_messages(topics=[ODOM_TOPIC], start_time=t_start+rospy.Duration(175), end_time=t_start+rospy.Duration(190)):
# # for _, msg_odom, t_odom in bag.read_messages(topics=[ODOM_TOPIC], start_time=t_start+rospy.Duration(30), end_time=t_start+rospy.Duration(50)):
#     x_velocity.append(msg_odom.twist.twist.linear.x)
# plt.plot(x_velocity)
# plt.show()
# print(np.mean(x_velocity))
# print(len(z_acceleration))


ax.scatter(z_acceleration, roll_rate, pitch_rate, c="blue", label="Grass")


# Name of the bag file
FILE = "bagfiles/raw_bagfiles/tom_road.bag"

bag = rosbag.Bag(FILE)
_, _, t_start = next(iter(bag.read_messages(topics=[ODOM_TOPIC])))

z_acceleration = []
roll_rate = []
pitch_rate = []

for _, msg_imu, t_imu in bag.read_messages(topics=[IMU_TOPIC], start_time=t_start+rospy.Duration(30), end_time=t_start+rospy.Duration(50)):
    z_acceleration.append(msg_imu.linear_acceleration.z)
    roll_rate.append(msg_imu.angular_velocity.x)
    pitch_rate.append(msg_imu.angular_velocity.y)

print(len(z_acceleration))

z_acceleration = z_acceleration - np.float32(9.81)

z_acceleration_f = rfft(z_acceleration)
roll_rate_f = rfft(roll_rate)
pitch_rate_f = rfft(pitch_rate)

plt.figure("Fourier road")
plt.subplot(131)
plt.plot(rfftfreq(len(z_acceleration), 1/IMU_SAMPLE_RATE),
         np.abs(z_acceleration_f))
plt.subplot(132)
plt.plot(rfftfreq(len(roll_rate), 1/IMU_SAMPLE_RATE),
         np.abs(roll_rate_f))
plt.subplot(133)
plt.plot(rfftfreq(len(pitch_rate), 1/IMU_SAMPLE_RATE),
         np.abs(pitch_rate_f))


ax.scatter(z_acceleration, roll_rate, pitch_rate, c="red", label="Road")


ax.set_xlabel("Z acceleration")
ax.set_ylabel("Roll rate")
ax.set_zlabel("Pitch rate")
ax.legend()

plt.show()
