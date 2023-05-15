import numpy as np


def energy_ratio(signal):
    return np.sum(signal**2)/len(signal)

def zero_crossing_rate(signal):
    return np.sum(np.abs(np.diff(np.sign(signal))))/(2*len(signal))

def waveform_length_ratio(signal):
    return np.sum(np.abs(np.diff(signal)))/len(signal)

def skewness(signal):
    return np.sum((signal-np.mean(signal))**3)/(len(signal)*np.std(signal)**3)

def kurtosis(signal):
    return np.sum((signal-np.mean(signal))**4)/(len(signal)*np.std(signal)**4)


if __name__ == "__main__":
    
    signal = np.array([1, -1, 6, -7, 1, -1, 1, -1])
    
    print(kurtosis(signal))