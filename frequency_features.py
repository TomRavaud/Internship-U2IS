import numpy as np


def spectral_centroid(magnitudes, frequencies):
    # Weighted mean of the frequencies present in the signal with their
    # magnitudes as the weights
    return np.sum(magnitudes*frequencies)/np.sum(magnitudes)

def spectral_spread(magnitudes, frequencies, sc):
    return np.sqrt(np.sum((frequencies - sc)**2 * magnitudes) / np.sum(magnitudes))

def spectral_energy(magnitudes):
    return np.sum(magnitudes**2)