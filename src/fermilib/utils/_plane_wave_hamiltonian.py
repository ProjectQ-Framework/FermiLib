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

"""Construct Hamiltonians in plan wave basis and its dual in 3D."""
from __future__ import absolute_import

import itertools

import numpy

from fermilib.config import *
from fermilib.ops import FermionOperator
from fermilib.utils._grid import Grid
from fermilib.utils._jellium import (orbital_id, grid_indices, position_vector,
                                     momentum_vector, jellium_model)
from fermilib.utils._molecular_data import periodic_hash_table
from fermilib.utils._grid import Grid

from projectq.ops import QubitOperator


def dual_basis_u_operator(grid, geometry, spinless):
    """Return the external potential operator in plane wave dual basis.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The dual basis operator.
    """
    prefactor = -4.0 * numpy.pi / grid.volume_scale()
    operator = None
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    for pos_indices in grid.all_points_indices():
        coordinate_p = position_vector(pos_indices, grid.length, grid.scale)
        for nuclear_term in geometry:
            coordinate_j = numpy.array(nuclear_term[1], float)
            for momenta_indices in grid.all_points_indices():
                momenta = momentum_vector(momenta_indices, grid.length,
                                          grid.scale)
                momenta_squared = momenta.dot(momenta)
                if momenta_squared < EQ_TOLERANCE:
                    continue
                exp_index = 1.0j * momenta.dot(coordinate_j - coordinate_p)
                coefficient = (prefactor / momenta_squared *
                    periodic_hash_table[nuclear_term[0]] * numpy.exp(exp_index))

                for spin_p in spins:
                    orbital_p = orbital_id(
                        grid.length, pos_indices, spin_p)
                    operators = ((orbital_p, 1), (orbital_p, 0))
                    if operator is None:
                        operator = FermionOperator(operators, coefficient)
                    else:
                        operator += FermionOperator(operators, coefficient)

    return operator


def plane_wave_u_operator(grid, geometry, spinless):
    """Return the external potential operator in plane wave basis.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        FermionOperator: The plane wave operator.
    """
    prefactor = -4.0 * numpy.pi / grid.volume_scale()
    operator = None
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    for indices_p in grid.all_points_indices():
        for indices_q in grid.all_points_indices():
            shift = grid.length // 2
            grid_indices_p_q = [
                (indices_p[i] - indices_q[i] + shift) % grid.length
                for i in range(grid.dimensions)]
            momenta_p_q = momentum_vector(grid_indices_p_q, grid.length,
                                          grid.scale)
            momenta_p_q_squared = momenta_p_q.dot(momenta_p_q)
            if momenta_p_q_squared < EQ_TOLERANCE:
                continue

            for nuclear_term in geometry:
                coordinate_j = numpy.array(nuclear_term[1])
                exp_index = 1.0j * momenta_p_q.dot(coordinate_j)
                coefficient = (prefactor / momenta_p_q_squared *
                    periodic_hash_table[nuclear_term[0]] * numpy.exp(exp_index))

                for spin in spins:
                    orbital_p = orbital_id(grid.length, indices_p, spin)
                    orbital_q = orbital_id(grid.length, indices_q, spin)
                    operators = ((orbital_p, 1), (orbital_q, 0))
                    if operator is None:
                        operator = FermionOperator(operators, coefficient)
                    else:
                        operator += FermionOperator(operators, coefficient)

    return operator


def plane_wave_hamiltonian(grid, geometry,
                           spinless=False, momentum_space=True):
    """Returns Hamiltonian as FermionOperator class.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.
        momentum_space (bool): Whether to return in plane wave basis (True)
            or plane wave dual basis (False).

    Returns:
        FermionOperator: The hamiltonian.
    """
    for item in geometry:
        if len(item[1]) != grid.dimensions:
            raise ValueError("Invalid geometry coordinate.")
        if item[0] not in periodic_hash_table:
            raise ValueError("Invalid nuclear element.")

    jellium_op = jellium_model(grid, spinless, momentum_space)

    if momentum_space:
        external_potential = plane_wave_u_operator(grid, geometry, spinless)
    else:
        external_potential = dual_basis_u_operator(grid, geometry, spinless)

    return jellium_op + external_potential


def fourier_transform(hamiltonian, n_dimensions, grid_length, length_scale,
                      spinless):
    """Apply Fourier tranform to change hamiltonian in plane wave basis.

    .. math::

        c^\dagger_v = \sqrt{1/N} \sum_m {a^\dagger_m \exp(-i k_v r_m)}
        c_v = \sqrt{1/N} \sum_m {a_m \exp(i k_v r_m)}

    Args:
        hamiltonian: The hamiltonian in plane wave basis.
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        hamiltonian_t: An instance of the FermionOperator class.
    """
    return _fourier_transform_helper(hamiltonian=hamiltonian,
                                     n_dimensions=n_dimensions,
                                     grid_length=grid_length,
                                     length_scale=length_scale,
                                     spinless=spinless,
                                     factor=+1,
                                     vec_func_1=momentum_vector,
                                     vec_func_2=position_vector)


def inverse_fourier_transform(hamiltonian, n_dimensions, grid_length,
                              length_scale, spinless):
    """Apply Fourier tranform to change hamiltonian in plane wave dual basis.

    .. math::

        a^\dagger_v = \sqrt{1/N} \sum_m {c^\dagger_m \exp(i k_v r_m)}
        a_v = \sqrt{1/N} \sum_m {c_m \exp(-i k_v r_m)}

    Args:
        hamiltonian: The hamiltonian in plane wave dual basis.
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        hamiltonian_t: An instance of the FermionOperator class.
    """
    return _fourier_transform_helper(hamiltonian=hamiltonian,
                                     n_dimensions=n_dimensions,
                                     grid_length=grid_length,
                                     length_scale=length_scale,
                                     spinless=spinless,
                                     factor=-1,
                                     vec_func_1=position_vector,
                                     vec_func_2=momentum_vector)


def _fourier_transform_helper(hamiltonian, n_dimensions, grid_length,
                              length_scale, spinless, factor,
                              vec_func_1, vec_func_2):
    hamiltonian_t = None

    for term in hamiltonian.terms:
        transformed_term = None
        for ladder_operator in term:
            indices_1 = grid_indices(ladder_operator[0], n_dimensions,
                                     grid_length, spinless)
            vec_1 = vec_func_1(indices_1, grid_length, length_scale)
            new_basis = None
            for indices_2 in itertools.product(range(grid_length),
                                               repeat=n_dimensions):
                vec_2 = vec_func_2(indices_2, grid_length, length_scale)
                if spinless:
                    spin = None
                else:
                    spin = ladder_operator[0] % 2
                orbital = orbital_id(grid_length, indices_2, spin)
                exp_index = factor * 1.0j * numpy.dot(vec_1, vec_2)
                if ladder_operator[1] == 1:
                    exp_index *= -1.0

                element = FermionOperator(((orbital, ladder_operator[1]),),
                                          numpy.exp(exp_index))
                if new_basis is None:
                    new_basis = element
                else:
                    new_basis += element

            new_basis *= numpy.sqrt(1.0/float(grid_length**n_dimensions))

            if transformed_term is None:
                transformed_term = new_basis
            else:
                transformed_term *= new_basis
        if transformed_term is None:
            continue

        # Coefficient.
        transformed_term *= hamiltonian.terms[term]

        if hamiltonian_t is None:
            hamiltonian_t = transformed_term
        else:
            hamiltonian_t += transformed_term

    return hamiltonian_t
