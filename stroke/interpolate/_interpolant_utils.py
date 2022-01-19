

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


def _linear_power(M, axis=0):
    """Take increasing power along input axis.

    Takes the ith power of the ith column elements if `axis=0`; takes the ith
    power of the ith row elements if `axis=1`.
    
    Parameters
    ----------
    M : np.ndarray
        2-D array.
    
    Returns
    -------
    Mi : np.ndarray
    """

    a1, a2 = M.shape

    if axis == 0:
        shape, a, b = (1, -1), a1, a2
    else:
        shape, a, b = (-1, 1), a2, a1

    p1 = np.arange(0, b, 1).reshape(shape)
    p2 = np.repeat(p1, a, axis=axis)

    return np.power(M, p2)
