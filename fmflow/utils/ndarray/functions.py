# coding: utf-8

# imported items
__all__ = ['rollrows']

# dependent packages
import numpy as np


# functions
def rollrows(array, shifts):
    """Roll 2D array elements of each row by a given shifts.

    Args:
        array (numpy.ndarray): A 2D array.
        shifts (int of list of int): The number(s) of places
            by which elements of each row are shifted.

    Returns:
        array (numpy.ndarray): An output rolled array.

    """
    array = np.asarray(array)
    shifts = np.asarray(shifts)

    if array.ndim < 2:
        return np.roll(array, shifts)

    if shifts.ndim < 1:
        shifts = np.tile(shifts, array.shape[0])

    rows, cols = np.ogrid[:array.shape[0], :array.shape[1]]
    cols = cols - (shifts%array.shape[1])[:,np.newaxis]
    cols[cols<0] += array.shape[1]
    return array[rows, cols]
