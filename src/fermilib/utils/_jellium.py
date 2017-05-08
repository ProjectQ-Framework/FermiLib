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

"""This module constructs Hamiltonians for the uniform electron gas."""
from __future__ import absolute_import

import itertools
import numpy

from fermilib.ops import FermionOperator

from projectq.ops import QubitOperator


# Exceptions.
class OrbitalSpecificationError(Exception):
    pass


def orbital_id(grid_length, grid_coordinates, spin=None):
    """Return the tensor factor of a orbital with given coordinates and spin.

    Args:
        grid_length: Int, the number of points in one dimension of the grid.
        grid_coordinates: List or tuple of ints giving coordinates of grid
            element. Acceptable to provide an int (instead of tuple or list)
            for 1D case.
        spin: Boole, 0 means spin down and 1 means spin up.
            If None, assume spinless model.

    Returns:
        tensor_factor: tensor factor associated with provided orbital label.

    Raises:
        OrbitalSpecificiationError: Invalid orbital coordinates provided.
    """
    # Initialize.
    if isinstance(grid_coordinates, int):
        grid_coordinates = [grid_coordinates]

    # Loop through dimensions of coordinate tuple.
    tensor_factor = 0
    for dimension, grid_coordinate in enumerate(grid_coordinates):

        # Make sure coordinate is an integer in the correct bounds.
        if isinstance(grid_coordinate, int) and grid_coordinate < grid_length:
            tensor_factor += grid_coordinate * (grid_length ** dimension)

        else:
            # Raise for invalid model.
            raise OrbitalSpecificationError(
                'Invalid orbital coordinates provided.')

    # Account for spin and return.
    if spin is None:
        return tensor_factor
    else:
        tensor_factor *= 2
        tensor_factor += spin
        return tensor_factor


def grid_indices(qubit_id, n_dimensions, grid_length, spinless):
    """This function is the inverse of orbital_id.

    Args:
        qubit_id: The tensor factor to map to grid indices.
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length (int): The number of points in one dimension of the grid.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        grid_indices: The location of the qubit on the grid.
    """
    # Remove spin degree of freedom.
    orbital_id = qubit_id
    if not spinless:
        if (orbital_id % 2):
            orbital_id -= 1
        orbital_id /= 2

    # Get grid indices.
    grid_indices = []
    for dimension in range(n_dimensions):
        remainder = orbital_id % (grid_length ** (dimension + 1))
        grid_index = remainder // (grid_length ** dimension)
        grid_indices += [grid_index]
    return grid_indices


def position_vector(position_indices, grid_length, length_scale):
    """Given grid point coordinate, return position vector with dimensions.

    Args:
        position_indices: List or tuple of integers giving grid point
            coordinate. Allowed values are ints in [0, grid_length).
        grid_length (int): The number of points in one dimension of the grid.
        length_scale (float): The real space length of a box dimension.

    Returns:
        position_vector: A numpy array giving the position vector with
        dimensions.

    Raises:
        orbitalSpecificationError: Position indices must be integers
            in [0, grid_length).
    """
    # Raise exceptions.
    if isinstance(position_indices, int):
        position_indices = [position_indices]
    if (not isinstance(grid_length, int) or
        max(position_indices) >= grid_length or
            min(position_indices) < 0.):
        raise orbitalSpecificationError(
            'Position indices must be integers in [0, grid_length).')

    # Compute position vector.
    shift = float(grid_length - 1) / 2.
    adjusted_vector = numpy.array(position_indices, float) - shift
    position_vector = length_scale * adjusted_vector / float(grid_length)
    return position_vector


def momentum_vector(momentum_indices, grid_length, length_scale):
    """Given grid point coordinate, return momentum vector with dimensions.

    Args:
        momentum_indices: List or tuple of integers giving momentum indices.
            Allowed values are ints in [0, grid_length).
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.

        Returns:
            momentum_vector: A numpy array giving the momentum vector with
                dimensions.

    Raises:
        OrbitalSpecificationError: Momentum indices must be integers
            in [0, grid_length).
    """
    # Raise exceptions.
    if isinstance(momentum_indices, int):
        momentum_indices = [momentum_indices]
    if (not isinstance(grid_length, int) or
        max(momentum_indices) >= grid_length or
            min(momentum_indices) < 0.):
        raise OrbitalSpecificationError(
            'Momentum indices must be integers in [0, grid_length).')

    # Compute momentum vector.
    shift = float(grid_length - 1) / 2.
    adjusted_vector = numpy.array(momentum_indices, float) - shift
    momentum_vector = 2. * numpy.pi * adjusted_vector / length_scale
    return momentum_vector


def momentum_kinetic_operator(n_dimensions, grid_length,
                              length_scale, spinless=False):
    """Return the kinetic energy operator in momentum second quantization.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    # Initialize.
    n_points = grid_length ** n_dimensions
    operator = FermionOperator()
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all plane waves.
    for grid_indices in itertools.product(range(grid_length),
                                          repeat=n_dimensions):
        momenta = momentum_vector(grid_indices, grid_length, length_scale)
        coefficient = momenta.dot(momenta) / 2.

        # Loop over spins.
        for spin in spins:
            orbital = orbital_id(grid_length, grid_indices, spin)

            # Add interaction term.
            operators = ((orbital, 1), (orbital, 0))
            operator += FermionOperator(operators, coefficient)

    return operator


def momentum_potential_operator(n_dimensions, grid_length,
                                length_scale, spinless=False):
    """Return the potential operator in momentum second quantization.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Boole, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.

    Raises:
        OrbitalSpecificationError: 'Must use an odd number of momentum modes.'
    """
    # Make sure number of orbitals is odd.
    if not (grid_length % 2):
        raise OrbitalSpecificationError(
            'Must use an odd number of momentum modes.')

    # Initialize.
    n_points = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    prefactor = 2. * numpy.pi / volume
    operator = FermionOperator((), 0.0)
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all plane waves.
    for omega_indices in itertools.product(range(grid_length),
                                           repeat=n_dimensions):
        shifted_omega_indices = [index - grid_length // 2 for
                                 index in omega_indices]

        # Get the momenta vectors.
        omega_momenta = momentum_vector(
            omega_indices, grid_length, length_scale)

        # Skip if omega momentum is zero.
        if not omega_momenta.any():
            continue

        # Compute coefficient.
        coefficient = prefactor / \
            omega_momenta.dot(omega_momenta)

        for grid_indices_a in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            shifted_indices_d = [
                (grid_indices_a[i] - shifted_omega_indices[i]) %
                grid_length for i in range(n_dimensions)]
            for grid_indices_b in itertools.product(range(grid_length),
                                                    repeat=n_dimensions):
                shifted_indices_c = [
                    (grid_indices_b[i] + shifted_omega_indices[i]) %
                    grid_length for i in range(n_dimensions)]

                # Loop over spins.
                for spin_a in spins:
                    orbital_a = orbital_id(
                        grid_length, grid_indices_a, spin_a)
                    orbital_d = orbital_id(
                        grid_length, shifted_indices_d, spin_a)
                    for spin_b in spins:
                        orbital_b = orbital_id(
                            grid_length, grid_indices_b, spin_b)
                        orbital_c = orbital_id(
                            grid_length, shifted_indices_c, spin_b)

                        # Add interaction term.
                        if (orbital_a != orbital_b) and \
                                (orbital_c != orbital_d):
                            operators = ((orbital_a, 1), (orbital_b, 1),
                                         (orbital_c, 0), (orbital_d, 0))
                            operator += FermionOperator(operators, coefficient)

    # Return.
    return operator


def position_kinetic_operator(n_dimensions, grid_length,
                              length_scale, spinless=False):
    """Return the kinetic operator in position space second quantization.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    # Initialize.
    n_points = grid_length ** n_dimensions
    operator = FermionOperator()
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all lattice sites.
    for grid_indices_a in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        coordinates_a = position_vector(
            grid_indices_a, grid_length, length_scale)
        for grid_indices_b in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            coordinates_b = position_vector(
                grid_indices_b, grid_length, length_scale)
            differences = coordinates_b - coordinates_a

            # Compute coefficient.
            coefficient = 0.
            for momenta_indices in itertools.product(range(grid_length),
                                                     repeat=n_dimensions):
                momenta = momentum_vector(
                    momenta_indices, grid_length, length_scale)
                if momenta.any():
                    coefficient += (
                        numpy.cos(momenta.dot(differences)) *
                        momenta.dot(momenta) / (2. * float(n_points)))

            # Loop over spins and identify interacting orbitals.
            for spin in spins:
                orbital_a = orbital_id(grid_length, grid_indices_a, spin)
                orbital_b = orbital_id(grid_length, grid_indices_b, spin)

                # Add interaction term.
                operators = ((orbital_a, 1), (orbital_b, 0))
                operator += FermionOperator(operators, coefficient)

    # Return.
    return operator


def position_potential_operator(n_dimensions, grid_length,
                                length_scale, spinless=False):
    """Return the potential operator in position space second quantization.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Boole, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    # Initialize.
    n_points = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    prefactor = 2. * numpy.pi / volume
    operator = FermionOperator()
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all lattice sites.
    for grid_indices_a in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        coordinates_a = position_vector(
            grid_indices_a, grid_length, length_scale)
        for grid_indices_b in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            coordinates_b = position_vector(
                grid_indices_b, grid_length, length_scale)
            differences = coordinates_b - coordinates_a

            # Compute coefficient.
            coefficient = 0.
            for momenta_indices in itertools.product(range(grid_length),
                                                     repeat=n_dimensions):
                momenta = momentum_vector(
                    momenta_indices, grid_length, length_scale)
                if momenta.any():
                    coefficient += (
                        prefactor * numpy.cos(momenta.dot(differences)) /
                        momenta.dot(momenta))

            # Loop over spins and identify interacting orbitals.
            for spin_a in spins:
                orbital_a = orbital_id(grid_length, grid_indices_a, spin_a)
                for spin_b in spins:
                    orbital_b = orbital_id(grid_length, grid_indices_b, spin_b)

                    # Add interaction term.
                    if orbital_a != orbital_b:
                        operators = ((orbital_a, 1), (orbital_a, 0),
                                     (orbital_b, 1), (orbital_b, 0))
                        operator += FermionOperator(operators, coefficient)

    return operator


def jellium_model(n_dimensions, grid_length, length_scale,
                  spinless=False, momentum_space=True):
    """Return jellium Hamiltonian as FermionOperator class.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.
        momentum_space: Boole, whether to return in momentum space (True)
            or position space (False).

    Returns:
        hamiltonian: An instance of the FermionOperator class.
    """
    if momentum_space:
        hamiltonian = momentum_kinetic_operator(n_dimensions,
                                                grid_length,
                                                length_scale,
                                                spinless)
        hamiltonian += momentum_potential_operator(n_dimensions,
                                                   grid_length,
                                                   length_scale,
                                                   spinless)
    else:
        hamiltonian = position_kinetic_operator(n_dimensions,
                                                grid_length,
                                                length_scale,
                                                spinless)
        hamiltonian += position_potential_operator(n_dimensions,
                                                   grid_length,
                                                   length_scale,
                                                   spinless)
    return hamiltonian


def jordan_wigner_position_jellium(n_dimensions, grid_length,
                                   length_scale, spinless=False):
    """Return the position space jellium Hamiltonian as QubitOperator.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        hamiltonian: An instance of the QubitOperator class.
    """
    # Initialize.
    n_orbitals = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    if spinless:
        spins = [None]
        n_qubits = n_orbitals
    else:
        spins = [0, 1]
        n_qubits = 2 * n_orbitals
    hamiltonian = QubitOperator()

    # Compute the identity coefficient.
    identity_coefficient = 0.
    for k_indices in itertools.product(range(grid_length),
                                       repeat=n_dimensions):
        momenta = momentum_vector(k_indices, grid_length, length_scale)
        if momenta.any():
            identity_coefficient += momenta.dot(momenta) / 2.
            identity_coefficient -= (numpy.pi * float(n_orbitals) /
                                     (momenta.dot(momenta) * volume))
    if spinless:
        identity_coefficient /= 2.

    # Add identity term.
    identity_term = QubitOperator((), identity_coefficient)
    hamiltonian += identity_term

    # Compute coefficient of local Z terms.
    z_coefficient = 0.
    for k_indices in itertools.product(range(grid_length),
                                       repeat=n_dimensions):
        momenta = momentum_vector(k_indices, grid_length, length_scale)
        if momenta.any():
            z_coefficient += numpy.pi / (momenta.dot(momenta) * volume)
            z_coefficient -= momenta.dot(momenta) / (4. * float(n_orbitals))

    # Add local Z terms.
    for qubit in range(n_qubits):
        qubit_term = QubitOperator(((qubit, 'Z'),), z_coefficient)
        hamiltonian += qubit_term

    # Add ZZ terms.
    prefactor = numpy.pi / volume
    for p in range(n_qubits):
        index_p = grid_indices(p, n_dimensions, grid_length, spinless)
        position_p = position_vector(index_p, grid_length, length_scale)
        for q in range(p + 1, n_qubits):
            index_q = grid_indices(q, n_dimensions, grid_length, spinless)
            position_q = position_vector(index_q, grid_length, length_scale)

            differences = position_p - position_q

            # Loop through momenta.
            zpzq_coefficient = 0.
            for k_indices in itertools.product(range(grid_length),
                                               repeat=n_dimensions):
                momenta = momentum_vector(k_indices, grid_length, length_scale)
                if momenta.any():
                    zpzq_coefficient += prefactor * numpy.cos(
                        momenta.dot(differences)) / momenta.dot(momenta)

            # Add term.
            qubit_term = QubitOperator(((p, 'Z'), (q, 'Z')), zpzq_coefficient)
            hamiltonian += qubit_term

    # Add XZX + YZY terms.
    prefactor = .25 / float(n_orbitals)
    for p in range(n_qubits):
        index_p = grid_indices(p, n_dimensions, grid_length, spinless)
        position_p = position_vector(index_p, grid_length, length_scale)
        for q in range(p + 1, n_qubits):
            if not spinless and (p + q) % 2:
                continue

            index_q = grid_indices(q, n_dimensions, grid_length, spinless)
            position_q = position_vector(index_q, grid_length, length_scale)

            differences = position_p - position_q

            # Loop through momenta.
            term_coefficient = 0.
            for k_indices in itertools.product(range(grid_length),
                                               repeat=n_dimensions):
                momenta = momentum_vector(k_indices, grid_length, length_scale)
                if momenta.any():
                    term_coefficient += prefactor * momenta.dot(momenta) * \
                        numpy.cos(momenta.dot(differences))

            # Add term.
            z_string = tuple((i, 'Z') for i in range(p + 1, q))
            xzx_operators = ((p, 'X'),) + z_string + ((q, 'X'),)
            yzy_operators = ((p, 'Y'),) + z_string + ((q, 'Y'),)
            hamiltonian += QubitOperator(xzx_operators, term_coefficient)
            hamiltonian += QubitOperator(yzy_operators, term_coefficient)

    # Return Hamiltonian.
    return hamiltonian
