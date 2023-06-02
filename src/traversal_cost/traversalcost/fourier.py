import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


def mean_filter(signal, window_size):
    """Apply a mean filter to a signal in order to reduce noise

    Args:
        signal (list): List of signal values
        window_size (int): Size of the window to apply the mean filter

    Returns:
        signal: Filtered signal
    """
    return uniform_filter1d(signal, size=window_size)

def hanning_window(signal):
    """Apply a hanning window to a signal

    Args:
        signal (list): List of signal values

    Returns:
        list: Signal with hanning window applied
    """
    # Apply a hanning window to the signal
    # (useful when the signal is not periodic)
    window = np.hanning(len(signal))
    
    return signal*window

def fft(signal, sample_rate, plot=False):
    """Compute the FFT of a signal

    Args:
        signal (list): List of signal values
        sample_rate (int): Sample rate of the signal
        plot (bool, optional): Whether to plot the FFT of the signal or not.
        Defaults to False.

    Returns:
        list, list: FFT of the signal, frequencies of the FFT
    """
    # Apply the FFT to the signal
    signal_f = rfft(signal)
    
    # Compute the frequencies of the FFT
    frequencies = rfftfreq(len(signal),
                           1/sample_rate)
    
    if plot:
        plt.figure()
        plt.plot(frequencies, np.abs(signal_f))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.title("FFT of signal")
        
    return signal_f, frequencies

def spectral_centroid(magnitudes, frequencies):
    """Compute the spectral centroid of a signal

    Args:
        magnitudes (list): FFT magnitudes of the signal
        frequencies (list): Frequencies of the FFT

    Returns:
        float: Spectral centroid of the signal
    """
    # Weighted mean of the frequencies present in the signal with their
    # magnitudes as the weights
    return np.sum(magnitudes*frequencies)/np.sum(magnitudes)

def spectral_spread(magnitudes, frequencies, sc):
    """Compute the spectral spread of a signal

    Args:
        magnitudes (list): FFT magnitudes of the signal
        frequencies (list): Frequencies of the FFT
        sc (float): Spectral centroid of the signal

    Returns:
        float: Spectral spread of the signal
    """    
    return np.sqrt(np.sum((frequencies-sc)**2*magnitudes)/np.sum(magnitudes))

def spectral_energy(magnitudes):
    """Compute the spectral energy of a signal

    Args:
        magnitudes (list): FFT magnitudes of the signal

    Returns:
        float: Spectral energy of the signal
    """
    return np.sum(magnitudes**2)

def spectral_roll_off(magnitudes, frequencies):
    """Compute the spectral roll-off of a signal

    Args:
        magnitudes (list): FFT magnitudes of the signal
        frequencies (list): Frequencies of the FFT

    Returns:
        float: Spectral roll-off of the signal
    """
    # Compute the spectral energy of the signal
    energy = spectral_energy(magnitudes)
    
    for i in range(len(frequencies)):
        if np.sum(magnitudes[:i]**2) >= 0.95*energy:
            return frequencies[i-1]
        
    return None
