import numpy as np


def variance(signal):
    """Compute the variance of a signal

    Args:
        signal (list): List of signal values

    Returns:
        ndarray (1,): Variance of the signal
    """
    var = np.var(signal)
    return np.array([var])
    
def energy_ratio(signal):
    """Compute the energy ratio of a signal

    Args:
        signal (list): List of signal values

    Returns:
        ndarray (1,): Energy ratio of the signal
    """
    er = np.sum(signal**2)/len(signal)
    return np.array([er])

def zero_crossing_rate(signal):
    """Compute the zero crossing rate of a signal

    Args:
        signal (list): List of signal values

    Returns:
        ndarray (1,): Zero crossing rate of the signal
    """
    zcr = np.sum(np.abs(np.diff(np.sign(signal))))/(2*len(signal))
    return np.array([zcr])

def waveform_length_ratio(signal):
    """Compute the waveform length ratio of a signal

    Args:
        signal (list): List of signal values

    Returns:
        ndarray(1,): Waveform length ratio of the signal
    """
    wlr = np.sum(np.abs(np.diff(signal)))/len(signal)
    return np.array([wlr])

def skewness(signal):
    """Compute the skewness of a signal

    Args:
        signal (list): List of signal values

    Returns:
        ndarray (1,): Skewness of the signal
    """
    skw = np.sum((signal-np.mean(signal))**3)/(len(signal)*np.std(signal)**3)
    return np.array([skw])

def kurtosis(signal):
    """Compute the kurtosis of a signal

    Args:
        signal (list): List of signal values

    Returns:
        ndarray (1,): Kurtosis of the signal
    """
    kur = np.sum((signal-np.mean(signal))**4)/(len(signal)*np.std(signal)**4)
    return np.array([kur])


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Create a dummy signal
    signal = np.array([1, -1, 6, -7, 1, -1, 1, -1])
    
    print(kurtosis(signal)) 
