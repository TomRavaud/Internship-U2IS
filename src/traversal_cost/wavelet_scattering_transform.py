import rosbag
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import numpy as np
import pandas as pd
import pywt

# import frequency_features as ff
# import time_features as tf

from kymatio.numpy import Scattering1D
from kymatio.datasets import fetch_fsdd


bag = rosbag.Bag("bagfiles/raw_bagfiles/speed_dependency/road.bag")

roll_velocity_values = []
pitch_velocity_values = []
vertical_acceleration_values = []

for topic, msg, t in bag.read_messages(topics=["/imu/data"]):
    roll_velocity_values.append(msg.angular_velocity.x)
    pitch_velocity_values.append(msg.angular_velocity.y)
    vertical_acceleration_values.append(msg.linear_acceleration.z - 9.81)

road = [[240, 610],
        [610, 1000],
        [1200, 1570],
        [1570, 1970],
        [2160, 2350],
        [2350, 2540],
        [2680, 2900],
        [2900, 3100],
        [3260, 3380],
        [3380, 3530],
        [3660, 3800],
        [3800, 3930],
        [4080, 4190],
        [4190, 4280],
        [4430, 4530],
        [4530, 4630],
        [4760, 4850],
        [4850, 4940],
        [5060, 5150],
        [5150, 5230]]

grass = [[150, 550],
        [550, 910],
        [1110, 1520],
        [1520, 1880],
        [2100, 2270],
        [2270, 2460],
        [2640, 2830],
        [2830, 3010],
        [3180, 3310],
        [3310, 3450],
        [3590, 3740],
        [3740, 3860],
        [4010, 4110],
        [4110, 4220],
        [4340, 4460],
        [4460, 4570],
        [4690, 4790],
        [4790, 4870],
        [5000, 5080],
        [5080, 5170]]

sand = [[220, 640],
        [640, 1000],
        [1200, 1620],
        [1620, 2000],
        [2170, 2360],
        [2360, 2560],
        [2730, 2950],
        [2950, 3110],
        [3280, 3420],
        [3420, 3530],
        [3710, 3830],
        [3830, 3950],
        [4100, 4210],
        [4210, 4300],
        [4450, 4560],
        [4560, 4650],
        [4780, 4880],
        [4880, 4970],
        [5080, 5180],
        [5180, 5260]]


# Create an array to store vibrational features
features = np.zeros((20, 882))

IMU_SAMPLE_RATE = 43


for i, segment in enumerate(road):

    start = segment[0]
    end = segment[1]

    # print("Segment ", i)
    
    # Get the signals
    roll_velocity_signal = roll_velocity_values[start:end]
    pitch_velocity_signal = pitch_velocity_values[start:end]
    vertical_acceleration_signal = vertical_acceleration_values[start:end]
    
    l = len(roll_velocity_signal)
    L = 420
    
    if l < L:
        pad = int(L - l)/2
        mode = "symmetric"

        roll_velocity_signal = pywt.pad(roll_velocity_signal, pad_widths=pad, mode=mode)
        pitch_velocity_signal = pywt.pad(pitch_velocity_signal, pad_widths=pad, mode=mode)
        vertical_acceleration_signal = pywt.pad(vertical_acceleration_signal, pad_widths=pad, mode=mode)
    
    # print(len(roll_velocity_signal))
    # print(len(pitch_velocity_signal))
    # print(len(vertical_acceleration_signal))
    # print("")
    
    # print("Signal length: ", len(signal))
    signal = pitch_velocity_signal
    
    #FIXME: to keep?
    signal = signal / np.max(np.abs(signal))

    T = signal.shape[-1]
    J = 6
    Q = 8

    scattering = Scattering1D(J, T, Q)

    Sx = scattering(signal)

    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    # print(T)
    # print(order2)
    # print(Sx[order0])
    # print(Sx[order1])
    # print(Sx[order2])
    # print(x)
    
    coefficients = np.vstack([Sx[order0], Sx[order1], Sx[order2]])
    # coefficients = np.vstack([Sx[order0], Sx[order1]])
    coefficients = coefficients.reshape(-1)
    
    features[i, :] = coefficients
    

np.save("src/traversal_cost/road.npy", features)
