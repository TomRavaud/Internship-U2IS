import numpy as np

# Import custom packages
import traversalcost.wavelets
import params.features


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

def dwt_variances(signal,
                  pad=False,
                  L=params.features.SIGNAL_LENGTH,
                  padding_mode=params.features.PADDING_MODE,
                  denoise=True,
                  wavelet=params.features.WAVELET,
                  nb_levels=params.features.NB_LEVELS,
                  threshold=params.features.DENOISE_THR,
                  denoising_mode=params.features.DENOISING_MODE):
    """Compute the variances of the coefficients of the discrete wavelet
    transform of a signal

    Args:
        signal (list): List of signal values

    Returns:
        ndarray (nb_levels+1,): Variances of the coefficients of the discrete
        wavelet transform of the signal
    """
    # Pad the signal
    if pad:
        signal = traversalcost.wavelets.pad_signal(signal,
                                                   L=L,
                                                   mode=padding_mode)
    
    # Apply discrete wavelet transform
    coefficients  = traversalcost.wavelets.dwt(signal,
                                               wavelet=wavelet,
                                               nb_levels=nb_levels)
    
    # De-noise the coefficients
    if denoise:
        coefficients = traversalcost.wavelets.denoise_signal(
            coefficients,
            threshold=threshold,
            mode=denoising_mode)
    
    # Compute the variances of the coefficients
    variances = np.array([np.var(c) for c in coefficients])
    
    return variances


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Create a dummy signal
    signal = np.random.rand(100)
    
    print(dwt_variances(signal).shape) 
