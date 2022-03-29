

import numpy as np
import numpy.typing as npt


def _extend_inst(inst1: list, n1: int, inst2: list, n2: int) -> list:
    """Combine instruction arrays.

    Parameters
    ----------
    inst1 : array
        1D array containing instruction arrays.
    n1 : int
        Length of `inst1`.
    inst2 : array
        1D array containing instruction arrays.
    n2 : int
        Length of `inst2`.

    Returns
    -------
    array
        1D array containing the combined instruction arrays.
    """

    for i in range(n2):

        opp, a, b, val = inst2[i][0], inst2[i][1], inst2[i][2], inst2[i][3]

        if a is not None:
            a += n1

        if b is not None:
            b += n1

        inst1.append([opp, a, b, val])

    return inst1


def _compute(inst: list, n: int, x: npt.ArrayLike, assume_ordered: bool) -> np.ndarray:
    """Recursively compute instructions.

    Parameters
    ----------
    inst : array
        1D array of instruction arrays.
    n : int
        Length of `inst`.
    x : array_like
        1D array or scalar values representing interpolation points.
    assume_ordered : bool
        Assumes interpolation points are ordered in increasing order if `True`.

    Returns
    -------
    np.ndarray
        1D array of the evaluated instruction set.
    """

    opp, val = inst[n][0], inst[n][3]

    if opp is None:
        return val(x, assume_ordered)
    elif inst[n][1] is None:
        a = val
        b = _compute(inst, inst[n][2], x, assume_ordered)
    elif inst[n][2] is None:
        a = _compute(inst, inst[n][1], x, assume_ordered)
        b = val
    else:
        a = _compute(inst, inst[n][1], x, assume_ordered)
        b = _compute(inst, inst[n][2], x, assume_ordered)

    temp = opp(a) if b is None else opp(a, b)

    return temp
