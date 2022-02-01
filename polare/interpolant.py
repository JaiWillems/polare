

from scipy.interpolate import interp1d
import numpy as np
import numpy.typing as npt


class Interp:
    """Interp(x, y, kind="linear")

    Interpolate 1-D array.

    `x` and `y` are arrays of values to describe a function ``y=f(x)`` for
    scalar ``x`` and ``y``. This class returns a function whose call method
    uses the specified interpolant order to find the value of new points.

    Parameters
    ----------
    x, y : array_like
        Arrays defining the data point coordinates and function outputs.

        The data in `x` define the independent variable; the data in `y`
        define the dependent variable.
    kind : {"linear", "quadratic", "cubic"}, optional
        The order of interpolation to use. Default is 'linear'.

    Attributes
    ----------
    _f : interp1d
        The interpolation callable.
    _kind : {"linear", "quadratic", "cubic"}
        The order of the interpolant.

    Methods
    -------
    __call__

    Notes
    -----
    The minimum number of data points required is ``d+1``, with ``d`` being the
    interpolant degree. If this condition is not satisfied, the degree will be
    set to ``n-1``, with ``n`` being the number of data points.

    Examples
    --------
    Construct a 1-D array and `Interp` object:

    >>> from stroke.interpolate import Interp
    >>> x = np.array([0, 2, 4, 6, 8, 10])
    >>> y = x ** 5
    >>> f = Interp(x=x, y=y, kind="quadratic")

    Now using the obtained interpolation function, plot the results:

    >>> import matplotlib.pyplot as plt
    >>> xnew = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ynew = f(xnew)
    >>> plt.plot(x, y, 'ro', xnew, ynew, 'b-')
    >>> plt.show()
    """

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, kind: str) -> None:

        self._f = interp1d(x, y, kind)
        self._kind = kind

    def __call__(self, x: npt.ArrayLike, assume_ordered: bool=False) -> np.ndarray:
        """Interpolate the function.

        Parameters
        ----------
        x : array_like
            1D array representing the x-coordinates on which to interpolate.
        assume_ordered : bool, optional
            Assumes interpolation points are ordered in increasing order if
            `True`.

        Returns
        -------
        y : array_like
            1D array representing the interpolated values.
        """

        xi = np.array(x, copy=True)

        if xi.ndim == 0:
            xi = np.array([x], copy=True)
        elif xi.ndim != 1:
            raise ValueError("x should be a scalar or 1D array.")

        if not assume_ordered:
            xi = np.sort(xi, kind="quicksort")

        return self._f(xi)
