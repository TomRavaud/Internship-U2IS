import rosbag
import rospy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rcParams['text.usetex'] = True  # Render Matplotlib text with Tex

from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d

import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd


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
data = np.load("samples/grassmud_t10_v1.npy")

z_acceleration = data[:, 0]
roll_rate = data[:, 1]
pitch_rate = data[:, 2]
x_velocity = data[:, 3]

IMU_SAMPLE_RATE = 43

# Apply a mean filter to reduce noise
roll_rate_mean = uniform_filter1d(roll_rate, size=5)

# Apply windowing because the signal is not periodic
hanning_window = np.hanning(len(z_acceleration))
roll_rate_windowing = roll_rate*hanning_window

roll_rate_mean_windowing = roll_rate_mean*hanning_window


# 2d time domain plots
plt.figure()

plt.subplot(221)
plt.plot(x_velocity)
plt.title("X velocitity")
plt.xlabel("Time ($s$)")
plt.ylabel("Velocity ($m/s$)")

plt.subplot(222)
plt.plot(z_acceleration)
plt.title("Z acceleration")
plt.xlabel("Time ($s$)")
plt.ylabel("Acceleration ($m/s^2$)")

plt.subplot(223)
plt.plot(roll_rate, "b", label="raw")
plt.plot(roll_rate_mean, "c", label="mean filter")
plt.plot(roll_rate_windowing, "m", label="Hanning windowing")
plt.plot(roll_rate_mean_windowing, "r", label="mean filter + Hanning windowing")
plt.title("Roll rate")
plt.xlabel("t")
plt.xlabel("Time ($s$)")
plt.ylabel("Angular rate ($rad/s$)")
plt.legend()

plt.subplot(224)
plt.plot(pitch_rate)
plt.title("Pitch rate")
plt.xlabel("Time ($s$)")
plt.ylabel("Angular rate ($rad/s$)")



z_acceleration_fourier = rfft(z_acceleration - 9.81)
roll_rate_fourier = rfft(roll_rate)
pitch_rate_fourier = rfft(pitch_rate)

roll_rate_mean_fourier = rfft(roll_rate_mean)
roll_rate_windowing_fourier = rfft(roll_rate_windowing)
roll_rate_mean_windowing_fourier = rfft(roll_rate_mean_windowing)

frequencies = rfftfreq(len(z_acceleration), 1/IMU_SAMPLE_RATE)


def spectral_centroid(magnitudes, frequencies):
    # Weighted mean of the frequencies present in the signal with their
    # magnitudes as the weights
    return np.sum(magnitudes*frequencies)/np.sum(magnitudes)

def spectral_spread(magnitudes, frequencies, sc):
    return np.sqrt(np.sum((frequencies - sc)**2 * magnitudes) / np.sum(magnitudes))

def spectral_energy(magnitudes):
    return np.sum(magnitudes**2)

def spectral_roll_off(magnitudes, frequencies):
    energy = spectral_energy(magnitudes)
    
    for i in range(len(frequencies)):
        if np.sum(magnitudes[:i]**2) >= 0.95*energy:
            return frequencies[i-1]
        
    return None


# 2d frequency domain plots
plt.figure()

plt.subplot(131)
plt.plot(frequencies,
         np.abs(z_acceleration_fourier))
plt.title("Z acceleration")
plt.xlabel("Frequency ($s^{-1}$)")
plt.ylabel("Amplitude ($m/s^2$)")

plt.subplot(132)
# plt.plot(frequencies,
#          np.abs(roll_rate_fourier), "b", label="raw")
# plt.axvline(spectral_centroid(np.abs(roll_rate_fourier), frequencies))

# plt.plot(frequencies,
#          np.abs(roll_rate_mean_fourier), "c", label="mean filter")
# plt.axvline(spectral_centroid(np.abs(roll_rate_mean_fourier), frequencies))

# plt.plot(frequencies,
#          np.abs(roll_rate_windowing_fourier), "m", label="Hanning windowing")
# plt.axvline(spectral_centroid(np.abs(roll_rate_windowing_fourier), frequencies))

plt.plot(frequencies,
         np.abs(roll_rate_mean_windowing_fourier), "r", label="mean filter + Hanning windowing")

sc = spectral_centroid(np.abs(roll_rate_mean_windowing_fourier), frequencies)
plt.axvline(sc, color="k", label="spectral centroid")

ss = spectral_spread(np.abs(roll_rate_mean_windowing_fourier), frequencies, sc)
print(ss)

sro = spectral_roll_off(np.abs(roll_rate_mean_windowing_fourier), frequencies)
# plt.axvline(sro)

plt.title("Roll rate")
plt.xlabel("Frequency ($s^{-1}$)")
plt.ylabel("Amplitude ($rad/s$)")
plt.legend()

plt.subplot(133)
plt.plot(frequencies,
         np.abs(pitch_rate_fourier))
plt.title("Pitch rate")
plt.xlabel("Frequency ($s^{-1}$)")
plt.ylabel("Amplitude ($rad/s$)")



# print("Roll rate variance: ", np.var(roll_rate))
# print("Roll rate mean variance: ", np.var(roll_rate_mean))
# print("Roll rate window variance: ", np.var(roll_rate_windowing))
# print("Roll rate mean window variance: ", np.var(roll_rate_mean_windowing))



# FILES = ["grass_t15_v1.npy",
#          "grasspath_t8_v1.npy",
#          "path_t6_v1.npy",
#          "road_t20_v1.npy",
#          "sidewalk_t5_v0.5.npy",
#          "pavedway_t3_v1.npy",
#          "grassmud_t10_v1.npy",
#          "grassleaves_t10_v0.5.npy",
#          "leaves_t10_v1.npy"]

# # 3d time domain plot and box plots
# fig = plt.figure("3D")
# ax = fig.add_subplot(projection="3d")

# fig5, ax5 = plt.subplots()
# fig6, ax6 = plt.subplots()
# fig7, ax7 = plt.subplots()

# labels = []
# z_accelerations = []
# roll_rates = []
# pitch_rates = []

# for file in FILES:
#     data = np.load("samples/" + file)
    
#     z_acceleration = data[:, 0]
#     roll_rate = data[:, 1]
#     pitch_rate = data[:, 2]
#     x_velocity  = data[:, 3]
    
#     label = file.split("_")[0]

#     ax.scatter(z_acceleration, roll_rate, pitch_rate, label=label)
    
#     labels.append(label)
#     z_accelerations.append(z_acceleration[:120])
#     roll_rates.append(roll_rate[:120])
#     pitch_rates.append(pitch_rate[:120])
    
    
#     z_acceleration_fourier = rfft(z_acceleration - 9.81)
#     roll_rate_fourier = rfft(roll_rate)
#     pitch_rate_fourier = rfft(pitch_rate)

#     frequencies = rfftfreq(len(z_acceleration), 1/IMU_SAMPLE_RATE)
    
#     ax5.plot(frequencies, np.abs(z_acceleration_fourier), label=label)
#     ax6.plot(frequencies, np.abs(roll_rate_fourier), label=label)
#     ax7.plot(frequencies, np.abs(pitch_rate_fourier), label=label)




# ax.set_xlabel("Z acceleration")
# ax.set_ylabel("Roll rate")
# ax.set_zlabel("Pitch rate")
# ax.legend()

# ax5.legend()
# ax6.legend()
# ax7.legend()

# fig2, ax2 = plt.subplots()
# ax2.boxplot(z_accelerations)
# ax2.set_xticklabels(labels)
# ax2.set_title("Z acceleration")

# fig3, ax3 = plt.subplots()
# ax3.boxplot(roll_rates)
# ax3.set_xticklabels(labels)
# ax3.set_title("Roll rate")

# fig4, ax4 = plt.subplots()
# ax4.boxplot(pitch_rates)
# ax4.set_xticklabels(labels)
# ax4.set_title("Pitch rate")


plt.show()


X = np.zeros((59, 12))
labels = []

directory = "samples/subsamples"

index_sample = 0

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    
    data = np.load(f)
    
    labels.append(filename.split("_")[0])
    
    for i in range(3):
        measurements = data[:, i] if i != 0 else data[:, i] - 9.81
    
        # Apply a mean filter to reduce noise
        measurements_mean = uniform_filter1d(measurements, size=5)
        
        # Compute the variance
        X[index_sample, i*4] = np.var(measurements_mean)

        # Apply windowing because the signal is not periodic
        hanning_window = np.hanning(50)
        measurements_mean_windowing = measurements_mean*hanning_window

        # Discrete Fourier transform
        measurements_mean_windowing_fourier = rfft(measurements)
        magnitudes = np.abs(measurements_mean_windowing_fourier)
        frequencies = rfftfreq(50, 1/IMU_SAMPLE_RATE)
        
        energy = spectral_energy(magnitudes)
        X[index_sample, i*4+1] = energy
        
        sc = spectral_centroid(magnitudes, frequencies)
        X[index_sample, i*4+2] = sc

        ss = spectral_spread(magnitudes, frequencies, sc)
        X[index_sample, i*4+3] = ss
    
    index_sample += 1
    

# print(labels)

# names = list(set(labels))
# print(names)

fig = plt.figure()
for i in range(len(labels)):
    plt.plot(X[i, 11], [0], "o", label=labels[i], markersize=10)
# plt.plot(X[:, 0], np.zeros_like(X[:, 0]), "r+")
plt.legend()
plt.title("Pitch rate spectral spread")

ax = fig.gca()

for i, p in enumerate(ax.get_lines()):    # this is the loop to change Labels and colors
    if p.get_label() in labels[:i]:    # check for Name already exists
        idx = labels.index(p.get_label())       # find ist index
        p.set_c(ax.get_lines()[idx].get_c())   # set color
        p.set_label('_' + p.get_label())  
plt.legend()


# Normalize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the dataset to a pandas DataFrame
columns = ["var z", "en z", "sc z", "ss z", "var r", "en r", "sc r", "ss r", "var p", "en p", "sc p", "ss p"]
dataframe = pd.DataFrame(X_scaled, columns=columns)

# Apply PCA
pca = PCA(n_components=2)
X_pc = pca.fit_transform(X_scaled)
dataframe_pc = pd.DataFrame(X_pc, columns=["pc1", "pc2"])

# print(dataframe_pc)


plt.figure()

labels_unique = list(set(labels))
labels = np.array(labels)

for label in labels_unique:
    indexes_label = labels == label
    plt.scatter(dataframe_pc.loc[indexes_label, 'pc1'],
                dataframe_pc.loc[indexes_label, 'pc2'],
                label=label)

plt.legend()
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.show()

# print(X_scaled)
# print(pca.components_)

# plt.figure()
# plt.plot(np.arange(pca.n_components_)+1, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Variance Explained')
plt.show()

