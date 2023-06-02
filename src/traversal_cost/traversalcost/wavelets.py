import numpy as np
import pywt
from kymatio.numpy import Scattering1D


def pad_signal(signal,
               L,
               mode):
    """Pad a signal to have a given length

    Args:
        signal (list): List of signal values
        L (int, optional): The length of the padded signal.
        Defaults to params.features.SIGNAL_LENGTH.
        mode (string, optional): The mode of the padding.
        Defaults to params.features.PADDING_MODE.

    Returns:
        list: The padded signal
    """
    # Get the length of the signal
    l = len(signal)
    
    # If the signal is too short, pad it
    if l < L:
        # Compute the number of elements to pad on each side
        pad = int(L - l)/2

        # Pad the signal
        signal = pywt.pad(signal, pad_widths=pad, mode=mode)
        
    elif l > L:
        print("The signal is too long.")
    
    return signal

def dwt(signal,
        wavelet,
        nb_levels):
    """Apply the discrete wavelet transform to a signal

    Args:
        signal (list): List of signal values
        wavelet (string, optional): The wavelet to use for the transform.
        Defaults to params.features.WAVELET.
        nb_levels (int, optional): The number of levels for the transform.
        Defaults to params.features.NB_LEVELS.

    Returns:
        list: The coefficients of the transform
    """
    # Apply discrete wavelet transform
    # approximation = coefficients[0]
    # detail = coefficients[1:]
    coefficients  = pywt.wavedec(signal,
                                 level=nb_levels,
                                 wavelet=wavelet)
    
    return coefficients

def denoise_signal(coefficients,
                   threshold,
                   mode):
    """De-noise a signal using the discrete wavelet transform

    Args:
        coefficients (list): The coefficients of the transform
        threshold (float, optional): The threshold for the de-noising.
        Defaults to params.features.DENOISE_THR.
        mode (string, optional): The mode for the de-noising.
        Defaults to params.features.DENOISING_MODE.

    Returns:
        list: The de-noised coefficients
    """    
    # De-noising (soft thresholding of the detail coefficients)
    for j in range(1, len(coefficients)):
        coefficients[j] = pywt.threshold(coefficients[j],
                                         value=threshold,
                                         mode=mode)
        
    return coefficients

def wst(signal,
        J=6,
        Q=8):
    """Apply the wavelet scattering transform to a signal

    Args:
        signal (list): List of signal values
        J (int, optional): 2**J is the averaging scale. Defaults to 6.
        Q (int, optional): Number of wavelets per octave. Defaults to 8.

    Returns:
        ndarray (n,): The scattering coefficients
    """    
    # Get the number of samples
    T = signal.shape[-1]

    # Define the wavelet scattering transform
    scattering = Scattering1D(J, T, Q)

    # Apply the wavelet scattering transform
    Sx = scattering(signal)

    # Extract the meta information from the scattering object and construct
    # masks for each order
    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    # Concatenate the scattering coefficients
    coefficients = np.vstack([Sx[order0], Sx[order1], Sx[order2]])
    coefficients = coefficients.reshape(-1)
    
    return coefficients


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Create a dummy signal
    signal = np.random.rand(500)
    
    print(wst(signal).shape)
