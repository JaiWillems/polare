

import numpy as np
from stroke.interpolate._interpolants import Vandermonde


class Interp:
    """Interp(x, y, kind="linear", method="poly")

    Interpolate 1-D array.

    `x` and `y` are arrays of values to describe a function ``y=f(x)`` for
    scalar ``x`` and ``y``. This class returns a function whose call method
    uses the specified interpolant to find the value of new points.

    Parameters
    ----------
    x, y : array_like
        Arrays defining the data point coordinates and function outputs.

        The data in `x` define the independent variable; the data in `y`
        define the dependent variable.
    kind : {"linear", "quadratic", "cubic"}, optional
        The order of interpolation to use. Default is 'linear'.
    method : {"poly"}, optional
        The type of interpolation. Default is 'poly'.

    Attributes
    ----------
    x, y : np.ndarray
        The input interpolation point data.
    x_min, x_max : float
        The minimum and maximum values of the independent array.
    k : {1, 2, 3}
        Interpolation degree.
    kind : {"linear", "quadratic", "cubic"}
        The order of the interpolant.
    method : {"poly"}
        Selected interpolation method.

    Methods
    -------
    __call__

    See Also
    --------
    Vandermonde : nth degree polynomial interpolation using Vandermonde.

    Notes
    -----
    The minimum number of data points required is ``d+1``, with ``d`` being the
    interpolant degree. If this condition is not satisfied, the degree will be
    set to ``n-1``, with ``n`` being the number of data points.

    Examples
    --------
    Construct a 1-D array and interpolate on it:

    >>> from stroke.interpolate import Interp
    >>> x = np.array([0, 2, 4, 6, 8, 10])
    >>> y = x ** 5
    >>> f = Interp(x=x, y=y, kind="quadratic", method="poly")

    Now use the obtained interpolation function to plot the result:

    >>> import matplotlib.pyplot as plt
    >>> xnew = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ynew = f(xnew)
    >>> plt.plot(x, y, 'ro', xnew, ynew, 'b-')
    >>> plt.show()
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, kind="linear", method="poly") -> None:

        interpolate_types = {"linear": 1, "quadratic": 2, "cubic": 3}
        try:
            self.k = interpolate_types[kind]
        except KeyError as e:
            raise ValueError(
                f"Unsupported interpolation type {repr(kind)}, must be "
                f"either {', '.join(map(repr, interpolate_types))}."
            ) from e

        method_types = {"poly": Vandermonde}
        try:
            self._f = method_types[method]
        except KeyError as e:
            raise ValueError(
                f"Unsupported interpolation method {repr(method)}, must be "
                f"either {', '.join(map(repr, method_types))}."
            ) from e

        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths.")

        self.x, self.y = [np.array(a, copy=True) for a in (x, y)]

        self.x_min = self.x.min()
        self.x_max = self.x.max()

        self.kind = kind
        self.method = method

    def __call__(self, x: np.ndarray, assume_ordered=False) -> np.ndarray:
        """Interpolate the function.

        Parameters
        ----------
        x : 1-D array
            The x-coordinates on which to interpolate.

        Returns
        -------
        y : 1-D array
            The interpolated values.
        """

        xi = np.array(x, copy=True)

        if xi.ndim != 1:
            raise ValueError("x should be a 1-D array.")

        if not assume_ordered:
            xi = np.sort(xi, kind="quicksort")

        out_of_bounds = np.any((xi < self.x_min) | (self.x_max < xi))
        if out_of_bounds:
            raise ValueError(r"Values out of range; x must be in %r"
                             % ((self.x_min, self.x_max),))

        return self._f(self.x, self.y, self.k, xi)
