"""Module for low-depth Trotterized simulation in the plane wave dual basis."""
import functools
import itertools
import numpy

from fermilib.circuits._parallel_bubble_sort import (
    index_of_position_in_1d_array, parallel_bubble_sort)
from fermilib.ops import FermionOperator
from fermilib.transforms import jordan_wigner
from fermilib.utils import count_qubits

from projectq.ops import (C, S, Sdag, X, Y, Z, H, CNOT, Swap, Rx, Ry, Rz, Ph,
                          All, Measure)


def fermionic_reorder(register, input_order, target_order=None):
    """Fermionically reorder the given register.

    Maps the input Jordan-Wigner order to the output order.

    Args:
        register (projectq.QuReg): Quantum register to reorder.
        input_order (list): The initial Jordan-Wigner canonical order.
        target_order (list): The desired Jordan-Wigner canonical order.
    """
    if not target_order:
        target_order = list(range(len(input_order)))

    key = (index_of_position_in_1d_array, [0])

    input_swaps = parallel_bubble_sort(list(itertools.product(input_order)),
                                       key, len(input_order))
    input_swaps = [swap[0] for swap_layer in input_swaps
                   for swap in swap_layer]

    output_swaps = parallel_bubble_sort(list(itertools.product(target_order)),
                                        key, len(input_order))
    output_swaps = [swap[0] for swap_layer in output_swaps
                    for swap in swap_layer]

    # Invert the output swaps to go from the input to the target.
    swaps = input_swaps + output_swaps[::-1]

    # Swap adjacent modes for all swaps in both rounds.
    for mode in swaps:
        C(Z) | (register[mode], register[mode + 1])
        Swap | (register[mode], register[mode + 1])


def special_F_adjacent(register, qubit_index, xx_yy_angle, zz_angle):
    """Apply the 'special F' (fermionic swap, XX+YY evolution, and ZZ
    evolution) gate to the register on qubit_index and qubit_index + 1.

    Args:
        register (projectq.QuReg): The register to apply the gate to.
        qubit_index (integer): The left qubit to act on.
        xx_yy_angle (float): The angle for evolution under XX+YY.
        zz_angle (float): The angle for evolution under ZZ.
    """
    Rx(numpy.pi / 2.) | register[qubit_index]
    CNOT | (register[qubit_index], register[qubit_index + 1])
    Rx(xx_yy_angle) | register[qubit_index]
    Ry(xx_yy_angle) | register[qubit_index + 1]
    CNOT | (register[qubit_index + 1], register[qubit_index])
    Rx(-numpy.pi / 2.) | register[qubit_index + 1]
    Rz(zz_angle) | register[qubit_index + 1]
    Sdag | register[qubit_index + 1]
    CNOT | (register[qubit_index], register[qubit_index + 1])
    S | register[qubit_index]
    S | register[qubit_index + 1]


def simulation_gate_trotter_step(register, hamiltonian, input_ordering=None,
                                 first_order=True):
    """Simulate a unit time Trotter step under the plane wave dual basis
    Hamiltonian using the fermionic simulation gate.

    Args:
        register (projectq.QuReg): The register to apply the unitary to.
        n_dimensions (int): The number of dimensions in the system.
        system_size (int): The side length along each dimension.
        input_ordering (list): The input Jordan-Wigner ordering.
        first_order (bool): Whether to apply a first or second-order
                            Trotter step.

    Notes:
        Applying a first-order Trotter step reverses the input ordering.
    """
    n_qubits = count_qubits(hamiltonian)

    if not input_ordering:
        input_ordering = list(range(n_qubits))

    # The intermediate ordering is the halfway point for second-order Trotter.
    intermediate_ordering = input_ordering[::-1]
    if first_order:
        final_ordering = intermediate_ordering
    else:
        final_ordering = input_ordering[:]

    # Whether we're in an odd or an even stagger. Alternates at each step.
    odd = 0

    # If we do a second order Trotter step, the input and final orderings
    # are the same. Use a flag to force us to leave the initial step.
    moved = False

    while input_ordering != final_ordering or not moved:
        moved = True
        for i in range(odd, n_qubits - 1, 2):
            left = input_ordering[i]
            right = input_ordering[i + 1]
            # For real c, c*([3^ 2] + [2^ 3]) transforms under JW to
            # c/2 (X_2 X_3 + Y_2 Y_3); evolution is exp(-ic/2(XX+YY))
            xx_yy_angle = hamiltonian.terms.get(((left, 1), (right, 0)), 0.0)

            left, right = max(left, right), min(left, right)
            # Two-body terms c 2^ 1^ 2 1 are equal to -c n_2 n_1. JW maps this
            # to -c(I-Z_2)(I-Z_1)/4 = -c I/4 + c Z_1/4 + c Z_2/4 - c Z_1 Z_2/4.
            # Evolution is thus exp(ic/4*(I - Z_1 - Z_2 + Z_1 Z_2)).
            zz_angle = -hamiltonian.terms.get(
                ((left, 1), (right, 1), (left, 0), (right, 0)), 0.0) / 2.

            # Divide by two for second order Trotter.
            if not first_order:
                xx_yy_angle /= 2
                zz_angle /= 2

            special_F_adjacent(register, i, xx_yy_angle, zz_angle)
            # The single-Z rotation angle is the opposite of the ZZ angle.
            Rz(-zz_angle) | register[i]
            Rz(-zz_angle) | register[i + 1]
            Ph(-zz_angle / 2.) | register

            num_operator_left = ((input_ordering[i], 1),
                                 (input_ordering[i], 0))
            if num_operator_left in hamiltonian.terms:
                # Jordan-Wigner maps a number term c*n_i to c*(I-Z_i)/2.
                # Time evolution is then exp(-i c*(I-Z_i)/2), which is equal to
                # a phase exp(-ic/2) and a rotation Rz(-c).
                z_angle = (-hamiltonian.terms[num_operator_left] /
                           (n_qubits - 1))

                # Divide by two for second order Trotter.
                if not first_order:
                    z_angle /= 2.
                Rz(z_angle) | register[i]
                Ph(z_angle / 2.) | register

            num_operator_right = ((input_ordering[i + 1], 1),
                                  (input_ordering[i + 1], 0))
            if num_operator_right in hamiltonian.terms:
                # Jordan-Wigner maps a number term c*n_i to c*(I-Z_i)/2.
                # Time evolution is then exp(-i c*(I-Z_i)/2), which is equal to
                # a phase exp(-ic/2) and a rotation Rz(-c).
                z_angle = (-hamiltonian.terms[num_operator_left] /
                           (n_qubits - 1))

                # Divide by two for second order Trotter.
                if not first_order:
                    z_angle /= 2
                Rz(z_angle) | register[i + 1]
                Ph(z_angle / 2.) | register

            # Finally, swap the two modes in input_ordering.
            input_ordering[i], input_ordering[i + 1] = (input_ordering[i + 1],
                                                        input_ordering[i])

        # Unless we're at the intermediate ordering, odd should flip.
        # This is only needed for second-order Trotter.
        if input_ordering != intermediate_ordering:
            odd = 1 - odd

    return input_ordering


def simulate_dual_basis_evolution(register, hamiltonian, trotter_steps=1,
                                  input_ordering=None, first_order=True):
    """Simulate Trotterized evolution under the plane wave Hamiltonian
    in the plane wave dual basis.

    Args:
        register (projectq.QuReg): The register to apply the unitary to.
        hamiltonian (FermionOperator): The normal-ordered dual basis
                                       Hamiltonian to simulate.
        trotter_steps (int): The number of Trotter steps to divide into.
        input_ordering (list): The input Jordan-Wigner ordering.
        first_order (bool): Whether to apply first or second-order
                            Trotter steps.

    Notes:
        Applying an odd number of first-order Trotter steps reverses
        the input ordering.
    """
    n_qubits = count_qubits(hamiltonian)

    if not input_ordering:
        input_ordering = list(range(n_qubits))

    if set(input_ordering) != set(range(n_qubits)):
        raise ValueError('input_ordering must be a permutation of integers '
                         '0 through the number of qubits.')

    if not isinstance(trotter_steps, int) or trotter_steps < 1:
        raise ValueError('The number of Trotter steps must be an int >0.')

    trotterized_hamiltonian = hamiltonian / float(trotter_steps)

    for i in range(trotter_steps):
        input_ordering = simulation_gate_trotter_step(
            register, trotterized_hamiltonian, input_ordering, first_order)

    return input_ordering
