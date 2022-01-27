

from stroke.interpolate import Interp
from stroke._stroke_utils import _extend_inst, _compute
from stroke._numpy_ufunc_overrides import HANDLED_FUNCTIONS
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

        return self._uniary_operation(np.positive)

    def __neg__(self):

        return self._uniary_operation(np.negative)

    def __add__(self, other):

        return self._binary_operation(np.add, other)

    def __radd__(self, other):

        return self._binary_operation(np.add, other, r=True)

    def __sub__(self, other):

        return self._binary_operation(np.subtract, other)

    def __rsub__(self, other):

        return self._binary_operation(np.subtract, other, r=True)

    def __mul__(self, other):

        return self._binary_operation(np.multiply, other)

    def __rmul__(self, other):

        return self._binary_operation(np.multiply, other, r=True)

    def __truediv__(self, other):

        return self._binary_operation(np.true_divide, other)

    def __rtruediv__(self, other):

        return self._binary_operation(np.true_divide, other, r=True)

    def __pow__(self, other):

        return self._binary_operation(np.power, other)

    def __rpow__(self, other):

        return self._binary_operation(np.power, other, r=True)

    def __abs__(self):

        return self._uniary_operation(np.abs)
    
    def _binary_operation(self, ufunc, other, r=False):
        """Return Stroke post binary operation.

        Parameters
        ----------
        ufunc : ufunc
            Binary NumPy universal function call.
        other : Stroke, int, float
            Second element in operation.
        r : bool, optional
            Represents a right operation when True.

        Returns
        -------
        Stroke
            Stroke post binary operation.
        """

        copy = self._copy()

        if isinstance(other, type(self)):
            copy._inst = _extend_inst(copy._inst, copy._n, other._inst, other._n)
            a, b, val = copy._n - 1, copy._n + other._n - 1, None
        else:
            a, b, val = copy._n - 1, None, other
        
        a, b, val = (b, a, val) if r else (a, b, val)

        copy._inst.append([ufunc, a, b, val])
        copy._n = len(copy._inst)

        return copy
    
    def _uniary_operation(self, ufunc):
        """Return Stroke post uniary operation.

        Parameters
        ----------
        ufunc : ufunc
            Uniary NumPy universal function call.

        Returns
        -------
        Stroke
            Stroke post uniary operation.
        """
        
        copy = self._copy()
        copy._inst.append([ufunc, copy._n - 1, None, None])
        copy._n += 1

        return copy

    def _handle_functions(self, func, *inputs):
        """Pre-process inputs to ufunc overrides.
        
        Parameters
        ----------
        func : function
            Overriding function.
        inputs : Stroke, int, float
            `func` inputs.
        
        Returns
        -------
        Stroke
            Post-processd Stroke.
        """

        copy = self._copy()
        i0, i1 = inputs[0], inputs[1]

        es, ev, xs, xv = None, None, None, None

        if isinstance(i0, (int, float)):
            ev, xs, n = i0, copy._n - 1, copy._n
        elif isinstance(i1, (int, float)):
            es, xv, n = copy._n - 1, i1, copy._n
        else:
            copy._inst = _extend_inst(i0._inst, i0._n, i1._inst, i1._n)
            copy._n = len(copy._inst)
            es, xs, n = i0._n - 1, i0._n + i1._n - 1, i0._n + i1._n

        new_inst = func(es, ev, xs, xv, n)
        copy._inst = _extend_inst(copy._inst, copy._n, new_inst, len(new_inst))
        copy._n = len(copy._inst)

        return copy    

    def _copy(self):
        """Copy Stroke object.

        Returns
        -------
        Stroke
            Stroke with different place in memory as original.
        """

        x, y = self._f.x.copy(), self._f.y.copy()
        kind, method = self._f.kind, self._f.method
        
        stroke_copy = Stroke(x, y, kind, method)
        stroke_copy._inst = self._inst.copy()
        stroke_copy._n = self._n

        return stroke_copy
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle NumPy universal functions.

        Parameters
        ----------
        ufunc : NumPy Universal Function
        method

        Returns
        -------
        Stroke
            Post-processed Stroke.
        """

        if method == '__call__':

            if ufunc in HANDLED_FUNCTIONS:
                func = HANDLED_FUNCTIONS[ufunc]
                return self._handle_functions(func, *inputs)

            try:

                i0, i1 = inputs[0], inputs[1]
                copy = self._copy()

                if isinstance(i0, (int, float)):
                    a, b, val = None, i1._n - 1, i0
                elif isinstance(i1, (int, float)):
                    a, b, val = i0._n - 1, None, i1
                else:
                    copy._inst = _extend_inst(i0._inst, i0._n, i1._inst, i1._n)
                    a, b, val = i0._n - 1, i0._n + i1._n - 1, None

            except:

                i0 = inputs[0]
                copy = self._copy()

                a, b, val = i0._n - 1, None, None

            copy._inst.append([ufunc, a, b, val])
            copy._n = len(copy._inst)

            return copy

        else:
            
            return NotImplemented
