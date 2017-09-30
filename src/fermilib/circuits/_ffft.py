#   Copyright 2017 ProjectQ-Framework (www.projectq.ch)
#
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

"""Functions to perform the fermionic fast Fourier transform."""
import itertools
import numpy

from fermilib.circuits._parallel_bubble_sort import (
    index_of_position_in_1d_array, parallel_bubble_sort)
from fermilib.ops import FermionOperator, normal_ordered
from fermilib.transforms import jordan_wigner
from fermilib.utils import fourier_transform, Grid

from projectq import MainEngine
from projectq.ops import (QubitOperator, H, X, Y, Z, C, Rx, Ry, Rz, Ph, Swap,
                          Measure, All, TimeEvolution)


def fswap_generator(mode_a, mode_b):
    """Give the two-qubit fermionic swap operator between two modes.

    Args:
        mode_a, mode_b (int): The two modes to be swapped.

    Returns:
        FermionOperator which swaps modes mode_a and mode_b.

    Notes:
        A fermionic swap is the same as the usual swap, with the
        difference that it applies a sign -1 to the state |11>.
    """
    return (FermionOperator('') +
            FermionOperator(((mode_a, 1), (mode_b, 0))) +
            FermionOperator(((mode_b, 1), (mode_a, 0))) -
            FermionOperator(((mode_a, 1), (mode_a, 0))) -
            FermionOperator(((mode_b, 1), (mode_b, 0))))


def fswap(register, mode_a, mode_b, fermion_to_spin_mapping=jordan_wigner):
    """Apply the fermionic swap operator to two modes.

    The fermionic swap is applied to the qubits corresponding to mode_a
    and mode_b, after the Jordan-Wigner transformation.

    Args:
        register (projectq.Qureg): The register of qubits to act on.
        mode_a, mode_b (int): The two modes to swap.
        fermion_to_spin_mapping (function): Transformation from fermionic
            to spin operators to use. Defaults to jordan_wigner.
    """
    operator = fswap_generator(mode_a, mode_b)
    TimeEvolution(numpy.pi / 2., fermion_to_spin_mapping(operator)) | register


def fswap_adjacent(register, mode):
    """Apply the fermionic swap operator to adjacent modes.

    The fermionic swap is applied to the qubits corresponding to mode
    and mode + 1, after the Jordan-Wigner transformation.

    Args:
        register (projectq.Qureg): The register of qubits to act on.
        mode (int): The lower of the two modes to apply the phase to.

    Notes:
        fswap_adjacent is a slightly optimized version of fswap for a
        more restricted case. Because the modes are adjacent, the
        Jordan-Wigner transform doesn't need to be computed.
    """
    operator = (QubitOperator(((mode, 'Z'),)) +
                QubitOperator(((mode + 1, 'Z'),)) +
                QubitOperator(((mode, 'X'), (mode + 1, 'X'))) +
                QubitOperator(((mode, 'Y'), (mode + 1, 'Y'))))
    TimeEvolution(numpy.pi / 4., operator) | register


def apply_phase(register, mode, phase):
    """Apply a phase to a fermionic mode if it is occupied.

    The phase is applied to the qubit corresponding to the mode under
    the Jordan-Wigner transformation.

    Args:
        register (projectq.Qureg): The register of qubits to act on.
        mode (int): The mode to apply the phase to.
    """
    Rz(2 * phase) | register[mode]


def fourier_transform_0_generator(mode_a, mode_b):
    """Give the generator of the two-qubit fermionic Fourier transform.

    Args:
        mode_a, mode_b (int): The two modes to Fourier transform.

    Notes:
      This is equivalent to a Hadamard transform on the singly-occupied
      space. It converts |01> to (|01> + |10>) / sqrt(2) and |10> to
      (|01> - |10>) / sqrt(2). Additionally, it applies a sign -1 to
      the state |11>. The state |00> is not changed.
    """
    xy = ((FermionOperator(((mode_a, 1),)) -
           1j * FermionOperator(((mode_a, 0),))) *
          (FermionOperator(((mode_b, 1),)) -
           1j * FermionOperator(((mode_b, 0),))))

    yx = ((FermionOperator(((mode_a, 1),)) +
           1j * FermionOperator(((mode_a, 0),))) *
          (FermionOperator(((mode_b, 1),)) +
           1j * FermionOperator(((mode_b, 0),))))

    return xy - yx


def fourier_transform_0(register, mode_a, mode_b):
    """Apply the fermionic Fourier transform to two modes.

    The fermionic Fourier transform is applied to the qubits
    corresponding to mode_a and mode_b, , after the Jordan-Wigner

    Args:
        register (projectq.Qureg): The register of qubits to act on.
        mode_a, mode_b (int): The two modes to Fourier transform.
    """
    operator = fourier_transform_0_generator(mode_a, mode_b)
    jw_operator = jordan_wigner(operator)
    Z | register[mode_b]
    TimeEvolution(numpy.pi / 8., jw_operator) | register


def fourier_transform_0_adjacent(register, mode):
    """Apply the fermionic Fourier transform to two adjacent modes.

    The fermionic Fourier transform is applied to the qubits
    corresponding to mode and mode + 1 after the Jordan-Wigner
    transformation.

    Args:
        register (projectq.Qureg): The register of qubits to act on.
        mode (int): The lower of the two modes to Fourier transform.
    """
    jw_operator = (QubitOperator(((mode, 'X'), (mode + 1, 'Y'))) -
                   QubitOperator(((mode, 'Y'), (mode + 1, 'X'))))
    Z | register[mode + 1]
    TimeEvolution(numpy.pi / 8., jw_operator) | register


def ffft(engine, register, n, start=0):
    """Apply the n-mode fermionic fast Fourier transform to the given
    register on qubits start to start + n (not inclusive).

    Args:
        engine (projectq.MainEngine): The simulator engine with the
                                      register.
        register (projectq.QuReg): The register to apply the FFFT to.
        n (int): The number of modes in the system.
        start (int): The starting mode for the FFFT. The FFFT acts on
                     modes start through start + n (not inclusive).
                     Defaults to 0.

    Notes:
        This algorithm uses radix-2 decimation-in-time, so n must be a
        binary power. This decimation is head recursive (the 2-mode FFFT
        is applied as the first part of the 4-mode FFFT, which is applied
        as the first part of the 8-mode FFFT, etc).
    """
    if n < 2 or (n & (n - 1)):
        raise ValueError('n must be a binary power.')

    if n == 2:
        fourier_transform_0_adjacent(register, start)
    else:
        ffft(engine, register, n // 2, start)
        ffft(engine, register, n // 2, start + n // 2)

        # Apply initial swap network.
        center = start + n // 2
        for i in range(n // 2):
            for j in range(i):
                # fswap is equivalent to Swap followed by C(Z)
                fswap_adjacent(register, center + 2*j - i)

        # Apply phases and two-qubit Fourier transforms.
        for i in range(n // 2):
            apply_phase(register, start + 2 * i + 1, -i * numpy.pi / n)
            fourier_transform_0_adjacent(register, start + 2 * i)

        # Undo the swap network.
        for i in range(n//2 - 1, -1, -1):
            for j in range(i):
                # fswap is equivalent to Swap followed by C(Z)
                fswap_adjacent(register, center + 2*j - i)


def ffft_2d(engine, register, system_size):
    """Apply the 2D fermionic fast Fourier transform to a register.

    Args:
        engine (projectq.MainEngine): The simulator engine with the
                                      register.
        register (projectq.QuReg): The register to apply the 2D FFFT to.
        system_size (int): The side length of the system. register must
                           thus have system_size ** 2 qubits.

    Notes:
        This algorithm uses radix-2 decimation-in-time, so system_size
        must be a binary power. This decimation is head recursive (the
        2-mode FFFT is applied as the first part of the 4-mode FFFT,
        which is applied as the first part of the 8-mode FFFT, etc).
    """
    # Apply the FFFT along one axis of the register.
    for i in range(system_size):
        ffft(engine, register[system_size*i: system_size*i + system_size],
             system_size)
        Ph(3 * numpy.pi / 4) | register[0]

    # To apply the FFFT along the second axis, we must fermionically
    # swap qubits into the correct positions. In 2D this is equivalent
    # to flipping all qubits across the "diagonal" of the grid.
    key_2d = (index_of_position_in_1d_array, (0, 1))
    arr = [(i, j) for i in range(system_size)
           for j in range(system_size)]
    swaps = parallel_bubble_sort(arr, key_2d, system_size)
    all_swaps = [swap for swap_layer in swaps for swap in swap_layer]

    # Fermionically swap into position to FFFT along the second axis.
    for swap in all_swaps:
        Swap | (register[swap[0]], register[swap[1]])
        C(Z) | (register[swap[0]], register[swap[1]])

    # Perform the FFFT along the second axis.
    for i in range(system_size):
        ffft(engine, register[system_size*i: system_size*i + system_size],
             system_size)
        Ph(3 * numpy.pi / 4) | register[0]

    # Undo the fermionic swap network to restore the original ordering.
    for swap in all_swaps[::-1]:
        Swap | (register[swap[0]], register[swap[1]])
        C(Z) | (register[swap[0]], register[swap[1]])


def swap_adjacent_fermionic_modes(fermion_operator, mode):
    """Swap adjacent modes in a fermionic operator.

    Returns: a new FermionOperator with mode and mode+1 swapped.

    Args:
        fermion_operator (projectq.FermionOperator): Original operator.
        mode (integer): The mode to be swapped with mode + 1.

    Notes:
        Because the swap must be fermionic, the sign of the operator is
        flipped if both creation operators mode^ and (mode+1)^ (or the
        corresponding annihilation operators) are present.
    """
    new_operator = FermionOperator.zero()

    for term in fermion_operator.terms:
        new_term = list(term)
        multiplier = 1
        if (mode, 1) in term and (mode + 1, 1) in term:
            multiplier *= -1
        else:
            if (mode, 1) in term:
                new_term[term.index((mode, 1))] = (mode + 1, 1)
            if (mode + 1, 1) in term:
                new_term[term.index((mode + 1, 1))] = (mode, 1)

        if (mode, 0) in term and (mode + 1, 0) in term:
            multiplier *= -1
        else:
            if (mode, 0) in term:
                new_term[term.index((mode, 0))] = (mode + 1, 0)
            if (mode + 1, 0) in term:
                new_term[term.index((mode + 1, 0))] = (mode, 0)

        new_operator.terms[tuple(new_term)] = (multiplier *
                                               fermion_operator.terms[term])

    return new_operator


def ffft_swap_networks(system_size, n_dimensions):
    """Compute the two networks of swaps required for the FFFT.

    Args:
        system_size (int): The system side length (the system is assumed
                           to be a hypercube).
        n_dimensions (int): Number of dimensions for the system.

    Notes:
        The reordering of the operators is the opposite of the ordering
        that would be applied to a state. This reordering is given in
        two stages. First, the bit-reversed output of the FFFT circuit
        (e.g. 0, 2, 1, 3 which is the four numbers 0, 1, 2, 3 after
        reversing their bit representations) is swapped to the standard
        order (e.g. 0, 1, 2, 3). Second, the standard order is swapped
        to the split-halves ordering used in fourier_transform (e.g.
        2, 3, 0, 1 which swaps the left and right halves of that range).

        Both rounds must be applied before the FFFT, but after the FFFT
        only the second round needs to be undone so as to restore to the
        order of the FFFT circuit. All reorderings must be done in both
        dimensions, and because the operator is fermionic all swaps must
        also be.
    """
    # Create a string formatting rule for converting a number into binary
    # using log2(system_size) digits.
    binary_format_rule = '{:0' + str(int(numpy.log2(system_size))) + 'b}'

    # Use the formatting rule to generate the bit-reversed output of the FFFT.
    bit_reversed_order = [int(binary_format_rule.format(i)[::-1], 2)
                          for i in range(system_size)]

    # Generate the split-halves order of fourier_transform.
    split_halves_order = (list(range(system_size // 2, system_size)) +
                          list(range(system_size // 2)))

    # Enumerate system coordinates.
    coordinates = list(itertools.product(*[range(system_size)
                                           for i in range(n_dimensions)]))

    # Create arrays of the grid coordinates corresponding to the two orders.
    # Within each dimension, the grid coordinates are in either bit-reversed
    # or split-halves ordering.
    bit_reversed_order_coordinates = [None] * system_size ** n_dimensions
    split_halves_order_coordinates = [None] * system_size ** n_dimensions
    for i in range(system_size ** n_dimensions):
        # e.g. for 4x4, this will be [(0, 0), (0, 2), (0, 1), (0, 3), (2, 0),
        # (2, 2), (2, 1), (2, 3), (1, 0), ...]; it is ordered along both dims
        # by the bit-reversed order 0, 2, 1, 3. The split-halves coordinates
        # are constructed similarly.
        bit_reversed_order_coordinates[i] = tuple(map(
            bit_reversed_order.__getitem__, coordinates[i]))
        split_halves_order_coordinates[i] = tuple(map(
            split_halves_order.__getitem__, coordinates[i]))

    # Find the two rounds of swaps for parallel bubble sort into
    # row-major order.
    key = (index_of_position_in_1d_array, range(n_dimensions - 1, -1, -1))
    first_round_swaps = parallel_bubble_sort(bit_reversed_order_coordinates,
                                             key, system_size)
    second_round_swaps = (
        parallel_bubble_sort(split_halves_order_coordinates,
                             key, system_size)[::-1])

    return [first_round_swaps, second_round_swaps]


def operator_2d_fft_with_reordering(fermion_operator, system_size):
    """Apply the 2D FFT to an operator after reordering its modes.

    Args:
        fermion_operator (projectq.FermionOperator): Original operator.
        system_size (int): The side length of the system. register must
                           thus have system_size ** 2 qubits.

    Notes:
        The reordering of the operators is the opposite of the ordering
        that would be applied to a state. This reordering is given in
        two stages. First, the bit-reversed output of the FFFT circuit
        (e.g. 0, 2, 1, 3 which is the four numbers 0, 1, 2, 3 after
        reversing their bit representations) is swapped to the standard
        order (e.g. 0, 1, 2, 3). Second, the standard order is swapped
        to the split-halves ordering used in fourier_transform (e.g.
        2, 3, 0, 1 which swaps the left and right halves of that range).

        Both rounds must be applied before the FFFT, but after the FFFT
        only the second round needs to be undone so as to restore to the
        order of the FFFT circuit. All reorderings must be done in both
        dimensions, and because the operator is fermionic all swaps must
        also be.
    """
    fermion_operator = normal_ordered(fermion_operator)
    grid = Grid(dimensions=2, length=system_size, scale=1.0)

    first_round_swaps, second_round_swaps = ffft_swap_networks(system_size,
                                                               n_dimensions=2)

    # Create the full list of swaps.
    swap_mode_list = first_round_swaps + second_round_swaps
    swaps = [swap[0] for swap_layer in swap_mode_list for swap in swap_layer]

    # Swap adjacent modes for all swaps in both rounds.
    for mode in swaps:
        fermion_operator = swap_adjacent_fermionic_modes(
            fermion_operator, mode)
        fermion_operator = normal_ordered(fermion_operator)

    # Apply the Fourier transform to the reordered operator.
    fft_result = fourier_transform(fermion_operator, grid, spinless=True)
    fft_result = normal_ordered(fft_result)

    # Undo the second round of swaps to restore the FFT's bit-reversed order.
    swap_mode_list = second_round_swaps
    swaps = [swap[0] for swap_layer in swap_mode_list for swap in swap_layer]
    for mode in swaps:
        fft_result = swap_adjacent_fermionic_modes(fft_result, mode)
        fft_result = normal_ordered(fft_result)

    return fft_result
