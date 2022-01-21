

from stroke.interpolate import Interp
from stroke._stroke_utils import _extend_inst, _compute
import numpy as np


class Stroke:
    """Stroke(x, y, kind="linear", method="poly")

    Kernel for continuous data operations.

    `x` and `y` are arrays of values to describe a function ``y=f(x)`` for
    scalar ``x`` and ``y``. This class stores continuous data approximations
    and alows for operations on the continuous approximation. The class can be
    called to find the value of new points.

    Parameters
    ----------
    x, y : array_like
        Arrays defining the data point coordinate and function output.

        The data in `x` defines the independent variable; the data in `y`
        define the dependent variable.
    kind : {"linear", "quadratic", "cubic"}, optional
        The order of interpolation to use. Default is 'linear'.
    method : {"poly"}, optional
        The type of interpolation. Default is 'poly'.

    Methods
    -------
    __call__

    Examples
    --------
    Construct a Stroke containing desired data:

    >>> from stroke import Stroke
    >>> x = np.linspace(-1, 1, 100)
    >>> y = np.exp(x) + np.cos(np.pi * x) + 1
    >>> s = Stroke(x=x, y=y, kind="quadratic", method="poly")

    Apply some operations:

    >>> 4 * (s + 2)

    Evaluate at any point in original `x` domain and plot:

    >>> import matplotlib.pyplot as plt
    >>> xnew = np.linspace(-1, 1, 1000)
    >>> ynew = s(xnew)
    >>> plt.plot(x, y, 'ro', xnew, ynew, 'b-')
    >>> plt.show()
    """

    def __init__(self, x, y, kind="linear", method="poly"):

        self._f = Interp(x, y, kind=kind, method=method)

        self._inst = [[None, None, None, self._f]]
        self._n = len(self._inst)

    def __call__(self, x):
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

        return _compute(self._inst, self._n - 1, x)

    def __pos__(self):

        return self

    def __neg__(self):

        self._inst.append([np.multiply, self._n - 1, None, -1])
        self._n += 1

        return self

    def __add__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n - 1, self._n + other._n - 1, None
        else:
            a, b, val = self._n - 1, None, other

        self._inst.append([np.add, a, b, val])
        self._n = len(self._inst)

        return self

    def __radd__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n + other._n - 1, self._n - 1, None
        else:
            a, b, val = None, self._n - 1, other

        self._inst.append([np.add, a, b, val])
        self._n = len(self._inst)

        return self

    def __sub__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n - 1, self._n + other._n - 1, None
        else:
            a, b, val = self._n - 1, None, other

        self._inst.append([np.subtract, a, b, val])
        self._n = len(self._inst)

        return self

    def __rsub__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n + other._n - 1, self._n - 1, None
        else:
            a, b, val = None, self._n - 1, other

        self._inst.append([np.subtract, a, b, val])
        self._n = len(self._inst)

        return self

    def __mul__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n - 1, self._n + other._n - 1, None
        else:
            a, b, val = self._n - 1, None, other

        self._inst.append([np.multiply, a, b, val])
        self._n = len(self._inst)

        return self

    def __rmul__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n + other._n - 1, self._n - 1, None
        else:
            a, b, val = None, self._n - 1, other

        self._inst.append([np.multiply, a, b, val])
        self._n = len(self._inst)

        return self

    def __truediv__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n - 1, self._n + other._n - 1, None
        else:
            a, b, val = self._n - 1, None, other

        self._inst.append([np.true_divide, a, b, val])
        self._n = len(self._inst)

        return self

    def __rtruediv__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n + other._n - 1, self._n - 1, None
        else:
            a, b, val = None, self._n - 1, other

        self._inst.append([np.true_divide, a, b, val])
        self._n = len(self._inst)

        return self

    def __pow__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n - 1, self._n + other._n - 1, None
        else:
            a, b, val = self._n - 1, None, other

        self._inst.append([np.power, a, b, val])
        self._n = len(self._inst)

        return self

    def __rpow__(self, other):

        if isinstance(other, type(self)):
            self._inst = _extend_inst(self._inst, self._n, other._inst, other._n)
            a, b, val = self._n + other._n - 1, self._n - 1, None
        else:
            a, b, val = None, self._n - 1, other

        self._inst.append([np.power, a, b, val])
        self._n = len(self._inst)

        return self

    def __abs__(self):

        self._inst.append([np.abs, self._n - 1, None, None])
        self._n += 1

        return self
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == '__call__':              

            if len(inputs) == 1:
                a, b, val = inputs[0]._n - 1, None, None
            elif len(inputs) == 2:
                self._inst = _extend_inst(inputs[0]._inst, inputs[0]._n, inputs[1]._inst, inputs[1]._n)
                a, b, val = inputs[0]._n - 1, inputs[0]._n + inputs[1]._n - 1, None

            self._inst.append([ufunc, a, b, val])
            self._n = len(self._inst)

            return self

        else:
            
            return NotImplemented

