import numpy as np

def lin_chirp(a, c=0, A=1,t=None, T=1, N=1024):
    """
    Computes N points of a linear chirp over [0,T] with phase Φ(t) = at + c/2 t².
    Formula is A(t) exp(i 2π Φ(t)).

    Parameters
    ----------
    a   (float)     :   First coefficient of the phase
    c   (float)     :   Second coefficient of the phase,                    optional (default is 0)
    A   (np.array)  :   Amplitude of the chirp,                             optional (default is 1)
    t   (np.array)  :   Domain of the signal,                               optional (default is None)
    T   (float)     :   Time period of the signal, ignored if t set         optional (default is 1)
    N   (int)       :   Size of the signal,                                 optional (default is 1024)

    Returns
    -------
    lin_chirp   (np.array)  :   the linear chirp
    t           (np.array)  :   Signal's support 

    """
    
    if t is None:
        t = np.arange(N)*T/N
    lin_chirp = A * np.exp(2j*np.pi*(a*t+c/2*t**2))
    return lin_chirp, t
