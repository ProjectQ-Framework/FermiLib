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

import numpy

from fermilib.config import *
from fermilib.ops import FermionOperator
from fermilib.utils._grid import Grid
from fermilib.utils._jellium import (orbital_id, grid_indices, position_vector,
                                     momentum_vector, jellium_model)
from fermilib.utils._molecular_data import periodic_hash_table


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


def fourier_transform(hamiltonian, grid, spinless):
    """Apply Fourier transform to change hamiltonian in plane wave basis.

    .. math::

        c^\dagger_v = \sqrt{1/N} \sum_m {a^\dagger_m \exp(-i k_v r_m)}
        c_v = \sqrt{1/N} \sum_m {a_m \exp(i k_v r_m)}

    Args:
        hamiltonian (FermionOperator): The hamiltonian in plane wave basis.
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The fourier-transformed hamiltonian.
    """
    return _fourier_transform_helper(hamiltonian=hamiltonian,
                                     grid=grid,
                                     spinless=spinless,
                                     phase_factor=+1,
                                     vec_func_1=momentum_vector,
                                     vec_func_2=position_vector)


def inverse_fourier_transform(hamiltonian, grid, spinless):
    """Apply inverse Fourier transform to change hamiltonian in plane wave dual
    basis.

    .. math::

        a^\dagger_v = \sqrt{1/N} \sum_m {c^\dagger_m \exp(i k_v r_m)}
        a_v = \sqrt{1/N} \sum_m {c_m \exp(-i k_v r_m)}

    Args:
        hamiltonian (FermionOperator):
            The hamiltonian in plane wave dual basis.
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The inverse-fourier-transformed hamiltonian.
    """

    return _fourier_transform_helper(hamiltonian=hamiltonian,
                                     grid=grid,
                                     spinless=spinless,
                                     phase_factor=-1,
                                     vec_func_1=position_vector,
                                     vec_func_2=momentum_vector)


def _fourier_transform_helper(hamiltonian,
                              grid,
                              spinless,
                              phase_factor,
                              vec_func_1,
                              vec_func_2):
    hamiltonian_t = FermionOperator.additive_identity()
    normalize_factor = numpy.sqrt(1.0 / float(grid.num_points()))

    for term in hamiltonian.terms:
        transformed_term = FermionOperator.multiplicative_identity()
        for ladder_op_mode, ladder_op_type in term:
            indices_1 = grid_indices(ladder_op_mode,
                                     grid.dimensions,
                                     grid.length,
                                     spinless)
            vec1 = vec_func_1(indices_1, grid.length, grid.scale)
            new_basis = FermionOperator.additive_identity()
            for indices_2 in grid.all_points_indices():
                vec2 = vec_func_2(indices_2, grid.length, grid.scale)
                spin = None if spinless else ladder_op_mode % 2
                orbital = orbital_id(grid.length, indices_2, spin)
                exp_index = phase_factor * 1.0j * numpy.dot(vec1, vec2)
                if ladder_op_type == 1:
                    exp_index *= -1.0

                element = FermionOperator(((orbital, ladder_op_type),),
                                          numpy.exp(exp_index))
                new_basis += element

            new_basis *= normalize_factor
            transformed_term *= new_basis

        # Coefficient.
        transformed_term *= hamiltonian.terms[term]

        hamiltonian_t += transformed_term

    return hamiltonian_t
