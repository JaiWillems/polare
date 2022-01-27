

import numpy as np


def _get_degree(n: int, d: int) -> int:
    return d if d < n else n - 1


def _get_number_segments(n: int, d: int) -> int:
    return (n - 1) // d + ((n - 1) % d > 0)


def _get_segments(x: np.ndarray, k: int) -> np.ndarray:
    """Return segment bounds for given `n` and `d`.
    
    Parameters
    ----------
    x : np.ndarray
    k : int
        Interpolant degree.
    
    Returns
    -------
    np.array
        Array containing segment bounds using values in `x`.
    """

    n = x.size
    ki = _get_degree(n, k)
    m = _get_number_segments(n, k)

    s = np.zeros((m, 2))
    i1 = np.arange(0, m, 1)

    i2, i3 = ki * i1, ki * (i1 + 1)
    i2[m - 1], i3[m - 1] = n - ki - 1, n - 1

    s[i1, 0], s[i1, 1] = x[i2], x[i3]

    return s


def _linear_power(M):
    """Take increasing power along the zero axis.
    
    Parameters
    ----------
    M : np.ndarray
        2-D array.
    
    Returns
    -------
    Mi : np.ndarray
    """

    [_, b] = M.shape
    e = np.arange(0, b, 1)

    return np.power(M, e)
