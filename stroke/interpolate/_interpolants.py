

import numpy as np
from stroke.interpolate._interpolant_utils import _get_segments, _linear_power


def Vandermonde(x: np.ndarray, y: np.ndarray, k: int, xi: np.ndarray) -> np.ndarray:
    """Interp(x, y, k, xi)

    Interpolate 1-D array.

    `x` and `y` are arrays of values to describe a function ``y=f(x)`` for
    scalar ``x`` and ``y``. The data is interpolated at the points ``xi``
    using Vandermonde polynomial interpolation.

    Parameters
    ----------
    x, y : np.ndarrays
        1-D arrays of equal length defining interpolation points.
    k : int
        Interpolant degree.
    xi : np.ndarray
        1-D array of points on which to interpolate.

    Returns
    -------
    yi : np.ndarray
        Interpolated points.

    Examples
    --------
    Construct a 1-D array and interpolate on it:

    >>> from stroke.interpolate._interpolants import Vandermonde
    >>> x = np.array([0, 2, 4, 6, 8, 10])
    >>> y = x ** 5
    >>> xi = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> yi = Vandermonde(x=x, y=y, k=2, xi=xi)

    Now plot the interpolated result:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'ro', xi, yi, 'b-')
    >>> plt.show()

    """

    s = _get_segments(x, k)

    yi = np.zeros(len(xi))
    for si in range(len(s)):

        i1 = np.where((s[si][0] <= x) & (x <= s[si][1]))[0]
        var = np.full((k + 1, x[i1].size), x[i1]).T
        var = _linear_power(var)

        c = np.linalg.solve(var, y[i1])

        i2 = np.where((s[si][0] <= xi) & (xi <= s[si][1]))[0]
        cc = np.full((len(i2), c.size), c)
        xixi = np.full((k + 1, xi[i2].size), xi[i2]).T
        xixi = _linear_power(xixi)

        yi[i2] = np.sum(cc * xixi, axis=1)

    return yi
