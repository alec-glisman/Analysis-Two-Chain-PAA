"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-14
Description: Helper functions to calculate statistical correlations
"""

# External Imports
import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """
    Calculates the autocorrelation function (ACF) of the input array.
    @SOURCE: https://scicoding.com/4-ways-of-calculating-autocorrelation-in-python/

    Parameters
    ----------
    x : np.ndarray
        Input array for autocorrelation

    Returns
    -------
    np.ndarray
        Autocorrelation function of input array
    """

    nt: int = len(x)
    size: int = 2 ** np.ceil(np.log2(2 * nt - 1)).astype("int")

    var: float = np.var(x)
    norm_x: np.ndarray = x - np.mean(x)

    fft: np.ndarray = np.fft.fft(norm_x, size)  # compute the FFT
    pwr: np.ndarray = np.abs(fft) ** 2  # get the power spectrum

    # Calculate the autocorrelation from inverse FFT of the power spectrum
    return np.fft.ifft(pwr).real[:nt] / (var * nt)
