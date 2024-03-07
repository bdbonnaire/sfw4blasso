import sys
sys.path.append("..")
import numpy as np
from py_lib.stft import stft

def gauss_spectrogram(f, std_dev, step_size=None, n_bin=None, prec=1e-3, return_window=False, **kwargs):
    """
    Computes the spectrogram of the 1D signal f with window g(x)= e^{-π x²/σ²}. 
    Discretization of the window follows that of the signal ; points g[k] kept
    are such that g[k]>= prec.

    Args:
    ----
        f           (1D np.array)               :   entry signal
        std_dev     (float)                     :   standard deviation σ of the gaussian window
        step_size   (float)                     :   step_size of the signal discretization, optional (default is 1/len(f))
        n_bins      (int)                       :   number of frequency bins, optional (default is len(f))
        prec        (int)                       :   defines the truncation of the window : g(x) >= prec, optional (default is 10^-3)

    Returns:
    -------
        spec        (n_bin x len(f) np.array)   :   The spectrogram of the given signal f
    """
    N = len(f)

    if step_size == None : step_size = 1/N
    if n_bin == None : n_bin = N

    # half length of the window to compute
    l = np.floor( (std_dev / step_size) * np.sqrt( -np.log(prec)/np.pi ) )
    T = np.arange(-l,l+1) * step_size
    win = np.exp(-np.pi * (T/std_dev)**2)
    spec = spectrogram(f, win, n_bin=n_bin, **kwargs)
    if return_window:
        return spec, win
    else:
        return spec

def spectrogram(f, window, n_bin=None, **kwargs):
    """
    Computes the spectrogram of f with given window

    Parameters
    ----------
    f       (1D np.array)               :   given signal
    window  (1D np.array)               :   window for the stft, len(window) must be smaller than len(f)
    n_bin   (int)                       :   number of frequency bins, optional (default is len(f))

    Returns
    -------
    spec    (n_bin x len(f) np.array)   :   spectrogram of f with given window

    """
    if n_bin == None : n_bin = len(f)

    kwargs_keys = ['cas', 'down', 'shift']
    for k in kwargs.keys():
        if k not in kwargs_keys:
            raise AttributeError(f"{k} is not an accepted stft argument. Accepted arguments are {kwargs_keys}")

    st, _ = stft(f,N=n_bin, h=window, **kwargs)
    spec = np.abs(st)**2
    return spec

def max_df_twoHarmonics(sigma, ratio=0.01):
    """
    Utility function when considering the spectrogram of two normalized harmonics, with gaussian window.
    Computes the maximal difference between the two frequencies until the interference pattern is considered visible.
    The interference pattern is considered visible when its max along the time axis in greater than ratio.
    Parameters
    ----------
    sigma   (float)     :   standard deviation of the gaussian window
    ratio   (float)     :   defines when the patters are considered visible, optional (default is 0.01)

    Returns
    -------
    (float) Max difference between frequencies for the interference pattern to be visible
    """ 
    return np.sqrt(-2/(np.pi*sigma**2) * np.log(ratio/4))
