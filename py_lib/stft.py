import numpy as np
from scipy.fft import fft

def stft(x, N, h, cas=1, down=1, shift=0):
    """
    Short Time Fourier Transform of a 1D signal

    Args:
        x (np.array)       Signal onto which to compute the stft
        N (int)             Number of frequency bins
        h (np.array)        Filter used as window
        cas (int)           value of 1, 2 or 3. respectively refers to equations (3), (6) and (9) in paper [1].  Defines the boundary methods and thus the reconstruction method.
        down (int)          Downsampling factor, accepted values lie between 1 and floor(length(h)/2)
        shift (int)         By how much to shift the signal

    Returns: 
        stft    short time fourier transform on x
        norm2h  L² norm of h on its support

[1] S. Meignen and D.-H. Pham, “Retrieval of the modes of multi- component signals from downsampled short-time Fourier transform,” IEEE Trans. Signal Process., vol. 66, no. 23, pp. 6204–6215, Dec.  2018.
    """
    Lh = len(h)//2
    if len(h) >= N : raise ValueError(f"Window too large : Size of window ({len(h)}) is greater than number of frequency bins ({N})")
    if down > Lh : raise ValueError("DownSampling factor is too large : Greater than half the filter")

    # Time window
    T = np.arange(shift, len(x), step=down)
    stft = np.zeros((N,len(T)), dtype=np.complex_)
    # norm of the (possibly truncated) filter
    norm2h = np.zeros(len(T))

    if cas == 1 : # No periodization, zero padding
        for t in T:
            # index indicates the uncentered indexes of the current window
            index = np.arange(-min(t, Lh), min(len(x)-t,Lh))
            stft[:len(index),t] = x[t+index] * h[index+Lh]
            stft[:,t] = fft(stft[:,t], n=N)
            # the L² norm of the truncated filter
            norm2h[t] = np.linalg.norm(h[index+Lh])
    elif cas in [2,3]: # periodization
        index = np.arange(-Lh, Lh+1)
        for t in T:
            stft[:len(index),t] = x[(t + index) % len(x)] * h
            stft[:,t] = fft(stft[:,t], n=N)
        norm2h.fill(np.linalg.norm(h))

    A = np.exp(2j * np.pi * np.arange(N)[:,None] @ T[None,:] / N)
    stft = A * stft
    return stft, norm2h
