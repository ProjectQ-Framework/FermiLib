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
from fermilib.utils._jellium import (orbital_id, grid_indices, position_vector,
                                     momentum_vector, jellium_model)

from projectq.ops import QubitOperator


def dual_basis_u_operator(n_dimensions, grid_length, length_scale,
                          nuclear_charges, spinless):
    """Return the external potential operator in plane wave dual basis.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        nuclear_charges: 3D int array, the nuclear charges.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    n_points = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    prefactor = -4.0 * numpy.pi / volume
    operator = None
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    for grid_indices_p in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        coordinate_p = position_vector(grid_indices_p, grid_length,
                                       length_scale)
        for grid_indices_j in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            coordinate_j = position_vector(grid_indices_j, grid_length,
                                           length_scale)
            for momenta_indices in itertools.product(range(grid_length),
                                                     repeat=n_dimensions):
                momenta = momentum_vector(momenta_indices, grid_length,
                                          length_scale)
                momenta_squred = momenta.dot(momenta)
                if momenta_squred < EQ_TOLERANCE:
                    continue
                exp_index = 1.0j * momenta.dot(coordinate_j - coordinate_p)
                coefficient = prefactor / momenta_squred * \
                    nuclear_charges[grid_indices_j] * numpy.exp(exp_index)

                for spin_p in spins:
                    orbital_p = orbital_id(
                        grid_length, grid_indices_p, spin_p)
                    operators = ((orbital_p, 1), (orbital_p, 0))
                    if operator is None:
                        operator = FermionOperator(operators, coefficient)
                    else:
                        operator += FermionOperator(operators, coefficient)

    return operator


def plane_wave_u_operator(n_dimensions, grid_length, length_scale,
                          nuclear_charges, spinless):
    """Return the external potential operator in plane wave basis.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        nuclear_charges: 3D int array, the nuclear charges.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    n_points = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    prefactor = -4.0 * numpy.pi / volume
    operator = None
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    for grid_indices_p in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        for grid_indices_q in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            shift = grid_length // 2
            grid_indices_p_q = [
                (grid_indices_p[i] - grid_indices_q[i] + shift) % grid_length
                for i in range(n_dimensions)]
            momenta_p_q = momentum_vector(grid_indices_p_q, grid_length,
                                          length_scale)
            momenta_p_q_squared = momenta_p_q.dot(momenta_p_q)
            if momenta_p_q_squared < EQ_TOLERANCE:
                continue

            for grid_indices_j in itertools.product(range(grid_length),
                                                    repeat=n_dimensions):
                coordinate_j = position_vector(grid_indices_j, grid_length,
                                               length_scale)
                exp_index = 1.0j * momenta_p_q.dot(coordinate_j)
                coefficient = prefactor / momenta_p_q_squared * \
                    nuclear_charges[grid_indices_j] * numpy.exp(exp_index)

                for spin in spins:
                    orbital_p = orbital_id(
                        grid_length, grid_indices_p, spin)
                    orbital_q = orbital_id(
                        grid_length, grid_indices_q, spin)
                    operators = ((orbital_p, 1), (orbital_q, 0))
                    if operator is None:
                        operator = FermionOperator(operators, coefficient)
                    else:
                        operator += FermionOperator(operators, coefficient)

    return operator


def plane_wave_hamiltonian(n_dimensions, grid_length, length_scale,
                           nuclear_charges, spinless=False,
                           momentum_space=True):
    """Returns Hamiltonian as FermionOperator class.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        nuclear_charges: 3D int array, the nuclear charges.
        spinless: Bool, whether to use the spinless model or not.
        momentum_space: Boole, whether to return in plane wave basis (True)
            or plane wave dual basis (False).

    Returns:
        hamiltonian: An instance of the FermionOperator class.
    """
    if len(nuclear_charges.shape) != n_dimensions:
        raise ValueError('Invalid nuclear charges array shape.')

    if momentum_space:
        return jellium_model(n_dimensions, grid_length, length_scale, spinless,
                             True) + \
            plane_wave_u_operator(n_dimensions, grid_length, length_scale,
                                  nuclear_charges, spinless)
    else:
        return jellium_model(n_dimensions, grid_length, length_scale, spinless,
                             False) + \
            dual_basis_u_operator(n_dimensions, grid_length, length_scale,
                                  nuclear_charges, spinless)


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
    hamiltonian_t = None

    for term in hamiltonian.terms:
        transformed_term = None
        for ladder_operator in term:
            momentum_indices = grid_indices(ladder_operator[0], n_dimensions,
                                            grid_length, spinless)
            momentum_vec = momentum_vector(momentum_indices, grid_length,
                                           length_scale)
            new_basis = None
            for position_indices in itertools.product(range(grid_length),
                                                      repeat=n_dimensions):
                position_vec = position_vector(position_indices, grid_length,
                                               length_scale)
                if spinless:
                    spin = None
                else:
                    spin = ladder_operator[0] % 2
                orbital = orbital_id(grid_length, position_indices, spin)
                exp_index = 1.0j * numpy.dot(momentum_vec, position_vec)
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
    hamiltonian_t = None

    for term in hamiltonian.terms:
        transformed_term = None
        for ladder_operator in term:
            position_indices = grid_indices(ladder_operator[0], n_dimensions,
                                            grid_length, spinless)
            position_vec = position_vector(position_indices, grid_length,
                                           length_scale)
            new_basis = None
            for momentum_indices in itertools.product(range(grid_length),
                                                      repeat=n_dimensions):
                momentum_vec = momentum_vector(momentum_indices, grid_length,
                                               length_scale)
                if spinless:
                    spin = None
                else:
                    spin = ladder_operator[0] % 2
                orbital = orbital_id(grid_length, momentum_indices, spin)
                exp_index = -1.0j * numpy.dot(position_vec, momentum_vec)
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
