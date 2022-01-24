

import numpy as np


HANDLED_FUNCTIONS = {}


def implements(np_function):
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.power)
def power(es, ev, xs, xv, n):
    """NumPy's `power` function override.

    Parameters
    ----------
    e : int
        Base instruction index.
    x : int
        Exponent instruction index.
    n : int
        Prior instruction length.

    Returns
    -------
    List
        Additional instructions to append.
    """

    if es is not None and xs is not None:

        inst = [[np.less, es - n, None, 0],
                [np.mod, xs - n, None, 2],
                [np.not_equal, 1, None, 0],
                [np.logical_and, 0, 2, None],
                [np.logical_not, 3, None, None],
                [np.multiply, 4, None, 2],
                [np.subtract, 5, None, 1],
                [np.absolute, es - n, None, None],
                [np.power, 7, xs - n, None],
                [np.multiply, 6, 8, None]]
    
    elif es is not None:

        val = (xv % 2) != 0
        inst = [[np.less, es - n, None, 0],
                [np.logical_and, 0, None, val],
                [np.logical_not, 1, None, None],
                [np.multiply, 2, None, 2],
                [np.subtract, 3, None, 1],
                [np.absolute, es - n, None, None],
                [np.power, 5, None, xv],
                [np.multiply, 4, 6, None]]
    
    else:

        val = ev < 0
        inst = [[np.mod, xs - n, None, 2],
                [np.not_equal, 0, None, 0],
                [np.logical_and, 1, None, val],
                [np.logical_not, 2, None, None],
                [np.multiply, 3, None, 2],
                [np.subtract, 4, None, 1],
                [np.power, None, xs - n, np.absolute(ev)],
                [np.multiply, 5, 6, None]]

    return inst


@implements(np.float_power)
def float_power(es, ev, xs, xv, n):
    """NumPy's `float_power` function override.

    Parameters
    ----------
    e : int
        Base instruction index.
    x : int
        Exponent instruction index.
    n : int
        Prior instruction length.

    Returns
    -------
    List
        Additional instructions to append.
    """

    if es is not None and xs is not None:

        inst = [[np.less, es - n, None, 0],
                [np.mod, xs - n, None, 2],
                [np.not_equal, 1, None, 0],
                [np.logical_and, 0, 2, None],
                [np.logical_not, 3, None, None],
                [np.multiply, 4, None, 2],
                [np.subtract, 5, None, 1],
                [np.absolute, es - n, None, None],
                [np.float_power, 7, xs - n, None],
                [np.multiply, 6, 8, None]]
    
    elif es is not None:

        val = (xv % 2) != 0
        inst = [[np.less, es - n, None, 0],
                [np.logical_and, 0, None, val],
                [np.logical_not, 1, None, None],
                [np.multiply, 2, None, 2],
                [np.subtract, 3, None, 1],
                [np.absolute, es - n, None, None],
                [np.float_power, 5, None, xv],
                [np.multiply, 4, 6, None]]
    
    else:

        val = ev < 0
        inst = [[np.mod, xs - n, None, 2],
                [np.not_equal, 0, None, 0],
                [np.logical_and, 1, None, val],
                [np.logical_not, 2, None, None],
                [np.multiply, 3, None, 2],
                [np.subtract, 4, None, 1],
                [np.float_power, None, xs - n, np.absolute(ev)],
                [np.multiply, 5, 6, None]]

    return inst
