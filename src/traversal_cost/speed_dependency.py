import rosbag
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
import numpy as np
import pandas as pd
import pywt

import traversalcost.fourier.frequency_features as ff


def compute_features(signal, rate):
    
    # Moving average
    signal_filtered = uniform_filter1d(signal, size=3)
    
    # Variance
    signal_variance = np.var(signal_filtered)
    
    # Apply windowing because the signal is not periodic
    hanning_window = np.hanning(len(signal))
    signal_mean_windowing = signal_filtered*hanning_window
    
    # Discrete Fourier transform
    signal_magnitudes = np.abs(rfft(signal_mean_windowing))
    frequencies = rfftfreq(len(signal), rate)
    
    # Energies
    signal_energy = ff.spectral_energy(signal_magnitudes)
    
    # Spectral centroids
    signal_sc = ff.spectral_centroid(signal_magnitudes, frequencies)
    
    # Spectral spread
    signal_ss = ff.spectral_spread(signal_magnitudes, frequencies, signal_sc)
    
    return np.array([signal_variance, signal_energy, signal_sc, signal_ss])


bag = rosbag.Bag("bagfiles/raw_bagfiles/speed_dependency/grass.bag")

roll_velocity_values = []
pitch_velocity_values = []
vertical_acceleration_values = []

for topic, msg, t in bag.read_messages(topics=["/imu/data"]):
    roll_velocity_values.append(msg.angular_velocity.x)
    pitch_velocity_values.append(msg.angular_velocity.y)
    vertical_acceleration_values.append(msg.linear_acceleration.z - 9.81)

# Visualize the data
# plt.figure()

# plt.subplot(131)
# plt.plot(roll_velocity_values, "b")

# plt.subplot(132)
# plt.plot(pitch_velocity_values, "r")

# plt.subplot(133)
# plt.plot(z_acceleration_values, "g")

# plt.show()


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


# Number of levels for the discrete wavelet transform
nb_levels = 2

# Create an array to store vibrational features
# features = np.zeros((20, 439))
features = np.zeros((20, 3*(nb_levels+1)))

IMU_SAMPLE_RATE = 43


for i, segment in enumerate(grass):

    start, end = segment

    # Get the signals
    roll_velocity_signal = roll_velocity_values[start:end]
    pitch_velocity_signal = pitch_velocity_values[start:end]
    vertical_acceleration_signal = vertical_acceleration_values[start:end]
    
    
    # Pad the signals to have the same length
    # l = len(roll_velocity_signal)
    # L = 420
    
    # if l < L:
    #     pad = int(L - l)/2
    #     mode = "symmetric"

    #     roll_velocity_signal = pywt.pad(roll_velocity_signal, pad_widths=pad, mode=mode)
    #     pitch_velocity_signal = pywt.pad(pitch_velocity_signal, pad_widths=pad, mode=mode)
    #     vertical_acceleration_signal = pywt.pad(vertical_acceleration_signal, pad_widths=pad, mode=mode)
    
    
    # Apply discrete wavelet transform
    # approximation = coefficients[0]
    # detail = coefficients[1:]
    wavelet = "db3"
    
    roll_coefficients  = pywt.wavedec(roll_velocity_signal,
                                 level=nb_levels,
                                 wavelet=wavelet)
    
    pitch_coefficients  = pywt.wavedec(pitch_velocity_signal,
                                 level=nb_levels,
                                 wavelet=wavelet)
    
    vertical_acceleration_coefficients  = pywt.wavedec(vertical_acceleration_signal,
                                 level=nb_levels,
                                 wavelet=wavelet)
     
    # wavelet = pywt.Wavelet('db2')
    # print(wavelet.dec_lo)
    # print(wavelet.dec_hi)
    # print(wavelet.dec_len)
    
    # coefficients = vertical_acceleration_coefficients
    
    # fig, axes = plt.subplots(len(coefficients)+1, 1)
    
    # axes[0].plot(roll_velocity_signal, "b+", label="Pitch rate")
    # axes[0].legend(loc="upper left")
    # axes[0].set_ylabel("Rate [rad/s]")
    
    # axes[1].plot(coefficients[0], "g-+", label="Approximation")
    # axes[1].legend(loc="upper left")
    # axes[1].set_ylabel("Rate [rad/s]")
   
    # for j in range(1, len(coefficients)):
        
    #     axes[j+1].plot(coefficients[j], "r-+", label="Detail - level " + str(j))
    #     axes[j+1].legend(loc="upper left")
    #     axes[j+1].set_ylabel("Rate [rad/s]")
        
    # plt.xlabel("Sample")
    # plt.show()
    
    
    # De-noising
    for j in range(1, len(roll_coefficients)):
        roll_coefficients[j] = pywt.threshold(roll_coefficients[j],
                                              value=0.005,
                                              mode="soft")
        pitch_coefficients[j] = pywt.threshold(pitch_coefficients[j],
                                               value=0.005,
                                               mode="soft")
        vertical_acceleration_coefficients[j] = pywt.threshold(vertical_acceleration_coefficients[j],
                                               value=0.005,
                                               mode="soft")
    
    
    # Fill the features array
    # features[i, 0:4] = compute_features(
    #     vertical_acceleration_signal,
    #     IMU_SAMPLE_RATE)
    # features[i, 4:8] = compute_features(
    #     roll_velocity_signal,
    #     IMU_SAMPLE_RATE)
    # features[i, 8:12] = compute_features(
    #     pitch_velocity_signal,
    #     IMU_SAMPLE_RATE)
    
    for j in range(len(roll_coefficients)):
        features[i, j] = np.var(roll_coefficients[j])
        
    for j in range(len(pitch_coefficients)):
        features[i, len(roll_coefficients)+j] = np.var(pitch_coefficients[j])
        
    for j in range(len(vertical_acceleration_coefficients)):
        features[i, len(roll_coefficients)+len(pitch_coefficients)+j] = np.var(vertical_acceleration_coefficients[j])


# Keep only features of interest
# features = features[:, [0, 2, 4, 6, 8, 10]]

# dataframe = pd.DataFrame(features, columns=["z_var", "z_sc", "roll_var", "roll_sc", "pitch_var", "pitch_sc"])
# print(dataframe)

np.save("src/traversal_cost/grass.npy", features)
