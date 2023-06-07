import numpy as np

# Import custom packages
import traversalcost.wavelets
import params.features
import params.robot
import traversalcost.fourier
import traversalcost.utils


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

def dwt_variances(
    signal: list,
    pad:bool =False,
    L: int=params.features.SIGNAL_LENGTH,
    padding_mode: str=params.features.PADDING_MODE,
    denoise: bool=True,
    wavelet: str=params.features.WAVELET,
    nb_levels: int=params.features.NB_LEVELS,
    threshold: float=params.features.DENOISE_THR,
    denoising_mode: str=params.features.DENOISING_MODE) -> np.ndarray:
    """Compute the variances of the coefficients of the discrete wavelet
    transform of a signal

    Args:
        signal (list): List of signal values
        pad (bool, optional): Whether to pad the signal or not. Defaults to
        False.
        L (int, optional): The length of the padded signal. Defaults to
        params.features.SIGNAL_LENGTH.
        padding_mode (str, optional): Padding mode for the signals that are too
        short. Defaults to params.features.PADDING_MODE.
        denoise (bool, optional): Whether to de-noise the coefficients or not.
        Defaults to True.
        wavelet (str, optional): Wavelet name. Defaults to
        params.features.WAVELET.
        nb_levels (int, optional): Number of levels for the discrete wavelet
        transform. Defaults to params.features.NB_LEVELS.
        threshold (float, optional): Threshold for the de-noising. Defaults to
        params.features.DENOISE_THR.
        denoising_mode (str, optional): De-noising mode. Defaults to
        params.features.DENOISING_MODE.

    Returns:
        np.ndarray (nb_levels+1,): Variances of the coefficients of the discrete
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

def spectral_centroid(
    signal: list,
    window_size: int=params.features.WINDOW_SIZE,
    sample_rate: float=params.robot.IMU_SAMPLE_RATE) -> np.ndarray:
    """Compute the spectral centroid of a signal

    Args:
        signal (list): List of signal values
        sample_rate (float, optional): The sample rate of the signal.
        Defaults to params.robot.IMU_SAMPLE_RATE.

    Returns:
        np.ndarray: The spectral centroid of the signal
    """
    # Apply a mean filter to the signal to reduce the noise
    signal = traversalcost.fourier.mean_filter(signal,
                                               window_size=window_size)
    
    # Apply windowing because the signal is not periodic
    signal = traversalcost.fourier.hanning_window(signal)
    
    # Compute the FFT of the signal
    signal_f, frequencies = traversalcost.fourier.fft(signal,
                                                      sample_rate=sample_rate)
    
    # Compute the spectral centroid
    sc = traversalcost.fourier.spectral_centroid(np.abs(signal_f),
                                                 frequencies)
    
    return np.array([sc])

def variance_and_spectral_centroid(
    signal: list,
    window_size: int=params.features.WINDOW_SIZE,
    sample_rate: float=params.robot.IMU_SAMPLE_RATE) -> np.ndarray:
    """Compute the variance and the spectral centroid of a signal

    Args:
        signal (list): List of signal values
        window_size (int, optional): The size of the window used in the mean
        filter. Defaults to params.features.WINDOW_SIZE.
        sample_rate (float, optional): The sample rate of the signal.
        Defaults to params.robot.IMU_SAMPLE_RATE.

    Returns:
        np.ndarray: The variance and the spectral centroid of the signal
    """    
    # Compute the variance of the signal
    var = variance(signal)
    
    # Compute the spectral centroid of the signal
    sc = spectral_centroid(signal,
                           window_size=window_size,
                           sample_rate=sample_rate)
    
    return np.concatenate((var, sc))

def dwt_variances_and_spectral_centroid(
    signal: list,
    window_size: int=params.features.WINDOW_SIZE,
    sample_rate: float=params.robot.IMU_SAMPLE_RATE,
    pad:bool =False,
    L: int=params.features.SIGNAL_LENGTH,
    padding_mode: str=params.features.PADDING_MODE,
    denoise: bool=True,
    wavelet: str=params.features.WAVELET,
    nb_levels: int=params.features.NB_LEVELS,
    threshold: float=params.features.DENOISE_THR,
    denoising_mode: str=params.features.DENOISING_MODE) -> np.ndarray:
    """Compute the variances of the coefficients of the discrete wavelet
    transform and the spectral centroid of a signal

    Args:
        signal (list): List of signal values
        window_size (int, optional): The size of the window used in the mean
        filter. Defaults to params.features.WINDOW_SIZE.
        sample_rate (float, optional): The sample rate of the signal.
        Defaults to params.robot.IMU_SAMPLE_RATE.
        pad (bool, optional): If True, the signal is padded. Defaults to False.
        L (int, optional): The length of the signal after padding.
        Defaults to params.features.SIGNAL_LENGTH.
        padding_mode (str, optional): The mode used for padding.
        Defaults to params.features.PADDING_MODE.
        denoise (bool, optional): If True, the signal is de-noised.
        Defaults to True.
        wavelet (str, optional): The wavelet used for the discrete wavelet
        transform. Defaults to params.features.WAVELET.
        nb_levels (int, optional): The number of levels of the discrete wavelet
        transform. Defaults to params.features.NB_LEVELS.
        threshold (float, optional): The threshold used for de-noising.
        Defaults to params.features.DENOISE_THR.
        denoising_mode (str, optional): The mode used for de-noising.
        Defaults to params.features.DENOISING_MODE.

    Returns:
        np.ndarray: The variances of the coefficients of the discrete wavelet
        transform and the spectral centroid of the signal
    """
    # Compute the variances of the coefficients of the discrete wavelet
    # transform
    variances = dwt_variances(signal,
                              pad=pad,
                              L=L,
                              padding_mode=padding_mode,
                              denoise=denoise,
                              wavelet=wavelet,
                              nb_levels=nb_levels,
                              threshold=threshold,
                              denoising_mode=denoising_mode)
    
    # Compute the spectral centroid
    sc = spectral_centroid(signal,
                           window_size=window_size,
                           sample_rate=sample_rate)
    
    return np.concatenate((variances, sc))

def wrapped_signal_fft(
    signal: list,
    N: int=params.features.N,
    window_size: int=params.features.WINDOW_SIZE,
    sample_rate: int=params.robot.IMU_SAMPLE_RATE) -> np.ndarray:
    """Compute the FFT of a wrapped signal

    Args:
        signal (list): List of signal values
        N (int, optional): The length of the block of the signal to wrap.
        Defaults to params.features.N.
        window_size (int, optional): The size of the window used in the mean
        filter. Defaults to params.features.WINDOW_SIZE.
        sample_rate (int, optional): The sample rate of the signal.
        Defaults to params.robot.IMU_SAMPLE_RATE.

    Returns:
        np.ndarray: _description_
    """
    # Wrap the signal
    signal = traversalcost.utils.modulo_wrap(signal,
                                             N=N)
    
    # Apply a mean filter to the signal to reduce the noise
    signal = traversalcost.fourier.mean_filter(signal,
                                               window_size=window_size)
    
    # Apply windowing because the signal is not periodic
    signal = traversalcost.fourier.hanning_window(signal)
    
    # Compute the FFT of the signal
    signal_f, _ = traversalcost.fourier.fft(signal,
                                            sample_rate=sample_rate)
    
    return np.abs(signal_f)


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Create a dummy signal
    signal = np.random.rand(100)
    
    print(wrapped_signal_fft(signal).shape)
