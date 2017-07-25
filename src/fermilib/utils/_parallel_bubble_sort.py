#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""This module has functions to perform parallel bubble sort on arrays.

Within the context of FermiLib, parallel bubble sort can be used to
quickly reorder fermionic modes or qubits.
"""
from functools import partial
from itertools import product


def index_of_position_in_1d_array(coordinate_ordering, system_side_length,
                                  position):
    """Return the index of an n-dimensional position embedded in a
    1-dimensional array.

    The index is computed using the given ordering of coordinate
    weights as well as the side length of the system from which the
    n-dimensional position is taken.

    Args:
        coordinate_ordering (list of ints): The order of coordinates by
                which to increment the index.
        system_side_length (int): The number of points along a side of
                the system.
        position (list of ints): The vector indicating the position
                within the n-dimensional system.

    Returns:
        The integer index of the position in a 1D array.

    Examples:
        index_of_position_in_1d_array((2, 1, 0), 4, (1, 2, 3))
                -> 27 (16 + 4 * 2 + 1 * 3).
        This is standard lexicographic ordering for 3D positions.

        index_of_position_in_1d_array((0, 1, 2), 2, (0, 1, 1))
                -> 6 (2 * 1 + 4 * 1)

        index_of_position_in_1d_array((2, 0, 1), 3, (2, 1, 0))
                -> 19 (9 * 2 + 1 + 3 * 0)
    """
    return sum(position[i] * system_side_length ** coordinate_ordering[i]
               for i in range(len(coordinate_ordering)))


def is_sorted_array_of_nd_positions(arr, key, system_side_length):
    """Determine whether an array of n-dim positions is sorted.

    Args:
        arr: the array to work with.
        key: the key function or tuple to use to determine ordering.
        system_side_length: The integer side length of the n-dimensional
                            cube from which the positions in the array
                            are taken.
    """
    if isinstance(key, tuple):
        key = partial(*key)

    for i in range(len(arr) - 1):
        if (key(system_side_length, arr[i]) >
                key(system_side_length, arr[i + 1])):
            return False
    return True


def parallel_bubble_sort(array, key, system_side_length):
    """Give the layers of swaps which sort an array of positions.

    Swaps can occur only between adjacent entries in the array.

    Args:
        array: An array of multi-dimensional positions to sort.
        key: The key function or tuple by which to determine ordering.
        system_side_length: The integer side length of the hypercube
                            from which the positions in array are taken.
    """
    swaps = []
    odd = 0

    if isinstance(key, tuple):
        key = partial(*key)

    while not is_sorted_array_of_nd_positions(array, key, system_side_length):
        swaps.append(parallel_bubble_sort_single_step(
            array, key, odd, system_side_length=system_side_length))
        odd = 1 - odd

    return swaps


def parallel_bubble_sort_single_step(array, key, odd=False,
                                     system_side_length=None):
    """Give a layer of swaps as part of a sorting network for an array of
    positions.

    Args:
        array: An array of multi-dimensional positions to sort.
        key: The key function by which to determine ordering.
        odd: Whether or not to swap all odd entries with their even
             neighbor (whether to the right or down), e.g. swapping
             x[i][j] with x[i+1][j] or x[i][j+1] with i or j even, which
             happens when odd = False, or with i or j odd, which happens
             when odd = True.
        system_side_length: The integer side length of the n-cube from
                            which the positions in array are taken.
    """
    swaps_in_layer = []

    for i in range(int(odd), len(array) - 1, 2):
        if (key(system_side_length, array[i]) >
                key(system_side_length, array[i + 1])):
            array[i], array[i+1] = array[i+1], array[i]
            swaps_in_layer.append((i, i + 1))

    return swaps_in_layer
