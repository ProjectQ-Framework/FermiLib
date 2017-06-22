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

import numpy
from projectq.ops import QubitOperator

from fermilib.ops import FermionOperator


# Exceptions.
class OrbitalSpecificationError(Exception):
    pass


def orbital_id(grid, grid_coordinates, spin=None):
    """Return the tensor factor of a orbital with given coordinates and spin.

    Args:
        grid (Grid): The discretization to use.
        grid_coordinates: List or tuple of ints giving coordinates of grid
            element. Acceptable to provide an int (instead of tuple or list)
            for 1D case.
        spin: Boole, 0 means spin down and 1 means spin up.
            If None, assume spinless model.

    Returns:
        tensor_factor (int):
            tensor factor associated with provided orbital label.
    """
    # Initialize.
    if isinstance(grid_coordinates, int):
        grid_coordinates = [grid_coordinates]

    # Loop through dimensions of coordinate tuple.
    tensor_factor = 0
    for dimension, grid_coordinate in enumerate(grid_coordinates):

        # Make sure coordinate is an integer in the correct bounds.
        if isinstance(grid_coordinate, int) and grid_coordinate < grid.length:
            tensor_factor += grid_coordinate * (grid.length ** dimension)

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


def grid_indices(qubit_id, grid, spinless):
    """This function is the inverse of orbital_id.

    Args:
        qubit_id (int): The tensor factor to map to grid indices.
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        grid_indices (numpy.ndarray[int]):
            The location of the qubit on the grid.
    """
    # Remove spin degree of freedom.
    orbital_id = qubit_id
    if not spinless:
        if (orbital_id % 2):
            orbital_id -= 1
        orbital_id /= 2

    # Get grid indices.
    grid_indices = []
    for dimension in range(grid.dimensions):
        remainder = orbital_id % (grid.length ** (dimension + 1))
        grid_index = remainder // (grid.length ** dimension)
        grid_indices += [grid_index]
    return grid_indices


def position_vector(position_indices, grid):
    """Given grid point coordinate, return position vector with dimensions.

    Args:
        position_indices (int|iterable[int]):
            List or tuple of integers giving grid point coordinate.
            Allowed values are ints in [0, grid_length).
        grid (Grid): The discretization to use.

    Returns:
        position_vector (numpy.ndarray[float])
    """
    # Raise exceptions.
    if isinstance(position_indices, int):
        position_indices = [position_indices]
    if not all(0 <= e < grid.length for e in position_indices):
        raise OrbitalSpecificationError(
            'Position indices must be integers in [0, grid_length).')

    # Compute position vector.
    adjusted_vector = numpy.array(position_indices, float) - grid.length // 2
    return grid.scale * adjusted_vector / float(grid.length)


def momentum_vector(momentum_indices, grid):
    """Given grid point coordinate, return momentum vector with dimensions.

    Args:
        momentum_indices: List or tuple of integers giving momentum indices.
            Allowed values are ints in [0, grid_length).
        grid (Grid): The discretization to use.

        Returns:
            momentum_vector: A numpy array giving the momentum vector with
                dimensions.
    """
    # Raise exceptions.
    if isinstance(momentum_indices, int):
        momentum_indices = [momentum_indices]
    if not all(0 <= e < grid.length for e in momentum_indices):
        raise OrbitalSpecificationError(
            'Momentum indices must be integers in [0, grid_length).')

    # Compute momentum vector.
    adjusted_vector = numpy.array(momentum_indices, float) - grid.length // 2
    return 2. * numpy.pi * adjusted_vector / grid.scale


def momentum_kinetic_operator(grid, spinless=False):
    """Return the kinetic energy operator in momentum second quantization.

    Args:
        grid (fermilib.utils.Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The kinetic momentum operator.
    """
    # Initialize.
    operator = FermionOperator()
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all plane waves.
    for momenta_indices in grid.all_points_indices():
        momenta = momentum_vector(momenta_indices, grid)
        coefficient = momenta.dot(momenta) / 2.

        # Loop over spins.
        for spin in spins:
            orbital = orbital_id(grid, momenta_indices, spin)

            # Add interaction term.
            operators = ((orbital, 1), (orbital, 0))
            operator += FermionOperator(operators, coefficient)

    return operator


def momentum_potential_operator(grid, spinless=False):
    """Return the potential operator in momentum second quantization.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        operator (FermionOperator)
    """
    # Initialize.
    volume = grid.volume_scale()
    prefactor = 2. * numpy.pi / volume
    operator = FermionOperator((), 0.0)
    spins = [None] if spinless else [0, 1]

    # Loop once through all plane waves.
    for omega_indices in grid.all_points_indices():
        shifted_omega_indices = [index - grid.length // 2 for
                                 index in omega_indices]

        # Get the momenta vectors.
        omega_momenta = momentum_vector(omega_indices, grid)

        # Skip if omega momentum is zero.
        if not omega_momenta.any():
            continue

        # Compute coefficient.
        coefficient = prefactor / omega_momenta.dot(omega_momenta)

        for grid_indices_a in grid.all_points_indices():
            shifted_indices_d = [
                (grid_indices_a[i] - shifted_omega_indices[i]) % grid.length
                for i in range(grid.dimensions)]
            for grid_indices_b in grid.all_points_indices():
                shifted_indices_c = [
                    (grid_indices_b[i] + shifted_omega_indices[i]) %
                    grid.length
                    for i in range(grid.dimensions)]

                # Loop over spins.
                for spin_a in spins:
                    orbital_a = orbital_id(grid, grid_indices_a, spin_a)
                    orbital_d = orbital_id(grid, shifted_indices_d, spin_a)
                    for spin_b in spins:
                        orbital_b = orbital_id(grid, grid_indices_b, spin_b)
                        orbital_c = orbital_id(grid, shifted_indices_c, spin_b)

                        # Add interaction term.
                        if ((orbital_a != orbital_b) and
                                (orbital_c != orbital_d)):
                            operators = ((orbital_a, 1), (orbital_b, 1),
                                         (orbital_c, 0), (orbital_d, 0))
                            operator += FermionOperator(operators, coefficient)

    # Return.
    return operator


def position_kinetic_operator(grid, spinless=False):
    """Return the kinetic operator in position space second quantization.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        operator (FermionOperator)
    """
    # Initialize.
    n_points = grid.num_points()
    operator = FermionOperator()
    spins = [None] if spinless else [0, 1]

    # Loop once through all lattice sites.
    for grid_indices_a in grid.all_points_indices():
        coordinates_a = position_vector(grid_indices_a, grid)
        for grid_indices_b in grid.all_points_indices():
            coordinates_b = position_vector(grid_indices_b, grid)
            differences = coordinates_b - coordinates_a

            # Compute coefficient.
            coefficient = 0.
            for momenta_indices in grid.all_points_indices():
                momenta = momentum_vector(momenta_indices, grid)
                if momenta.any():
                    coefficient += (
                        numpy.cos(momenta.dot(differences)) *
                        momenta.dot(momenta) / (2. * float(n_points)))

            # Loop over spins and identify interacting orbitals.
            for spin in spins:
                orbital_a = orbital_id(grid, grid_indices_a, spin)
                orbital_b = orbital_id(grid, grid_indices_b, spin)

                # Add interaction term.
                operators = ((orbital_a, 1), (orbital_b, 0))
                operator += FermionOperator(operators, coefficient)

    # Return.
    return operator


def position_potential_operator(grid, spinless=False):
    """Return the potential operator in position space second quantization.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        operator (FermionOperator)
    """
    # Initialize.
    volume = grid.volume_scale()
    prefactor = 2. * numpy.pi / volume
    operator = FermionOperator()
    spins = [None] if spinless else [0, 1]

    # Loop once through all lattice sites.
    for grid_indices_a in grid.all_points_indices():
        coordinates_a = position_vector(grid_indices_a, grid)
        for grid_indices_b in grid.all_points_indices():
            coordinates_b = position_vector(grid_indices_b, grid)
            differences = coordinates_b - coordinates_a

            # Compute coefficient.
            coefficient = 0.
            for momenta_indices in grid.all_points_indices():
                momenta = momentum_vector(momenta_indices, grid)
                if momenta.any():
                    coefficient += (
                        prefactor * numpy.cos(momenta.dot(differences)) /
                        momenta.dot(momenta))

            # Loop over spins and identify interacting orbitals.
            for spin_a in spins:
                orbital_a = orbital_id(grid, grid_indices_a, spin_a)
                for spin_b in spins:
                    orbital_b = orbital_id(grid, grid_indices_b, spin_b)

                    # Add interaction term.
                    if orbital_a != orbital_b:
                        operators = ((orbital_a, 1), (orbital_a, 0),
                                     (orbital_b, 1), (orbital_b, 0))
                        operator += FermionOperator(operators, coefficient)

    return operator


def jellium_model(grid, spinless=False, momentum_space=True):
    """Return jellium Hamiltonian as FermionOperator class.

    Args:
        grid (fermilib.utils.Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        momentum_space (bool): Whether to return in momentum space (True)
            or position space (False).

    Returns:
        FermionOperator: The Hamiltonian of the model.
    """
    if grid.length & 1 == 0 and grid.length & (grid.length - 1):
        raise OrbitalSpecificationError(
            'Must use an odd number or a power of 2 for momentum modes.')

    if momentum_space:
        hamiltonian = momentum_kinetic_operator(grid, spinless)
        hamiltonian += momentum_potential_operator(grid, spinless)
    else:
        hamiltonian = position_kinetic_operator(grid, spinless)
        hamiltonian += position_potential_operator(grid, spinless)
    return hamiltonian


def jordan_wigner_position_jellium(grid, spinless=False):
    """Return the position space jellium Hamiltonian as QubitOperator.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        hamiltonian (QubitOperator)
    """
    # Initialize.
    n_orbitals = grid.num_points()
    volume = grid.volume_scale()
    if spinless:
        n_qubits = n_orbitals
    else:
        n_qubits = 2 * n_orbitals
    hamiltonian = QubitOperator()

    # Compute the identity coefficient.
    identity_coefficient = 0.
    for k_indices in grid.all_points_indices():
        momenta = momentum_vector(k_indices, grid)
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
    for k_indices in grid.all_points_indices():
        momenta = momentum_vector(k_indices, grid)
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
        index_p = grid_indices(p, grid, spinless)
        position_p = position_vector(index_p, grid)
        for q in range(p + 1, n_qubits):
            index_q = grid_indices(q, grid, spinless)
            position_q = position_vector(index_q, grid)

            differences = position_p - position_q

            # Loop through momenta.
            zpzq_coefficient = 0.
            for k_indices in grid.all_points_indices():
                momenta = momentum_vector(k_indices, grid)
                if momenta.any():
                    zpzq_coefficient += prefactor * numpy.cos(
                        momenta.dot(differences)) / momenta.dot(momenta)

            # Add term.
            qubit_term = QubitOperator(((p, 'Z'), (q, 'Z')), zpzq_coefficient)
            hamiltonian += qubit_term

    # Add XZX + YZY terms.
    prefactor = .25 / float(n_orbitals)
    for p in range(n_qubits):
        index_p = grid_indices(p, grid, spinless)
        position_p = position_vector(index_p, grid)
        for q in range(p + 1, n_qubits):
            if not spinless and (p + q) % 2:
                continue

            index_q = grid_indices(q, grid, spinless)
            position_q = position_vector(index_q, grid)

            differences = position_p - position_q

            # Loop through momenta.
            term_coefficient = 0.
            for k_indices in grid.all_points_indices():
                momenta = momentum_vector(k_indices, grid)
                if momenta.any():
                    term_coefficient += (prefactor *
                                         momenta.dot(momenta) *
                                         numpy.cos(momenta.dot(differences)))

            # Add term.
            z_string = tuple((i, 'Z') for i in range(p + 1, q))
            xzx_operators = ((p, 'X'),) + z_string + ((q, 'X'),)
            yzy_operators = ((p, 'Y'),) + z_string + ((q, 'Y'),)
            hamiltonian += QubitOperator(xzx_operators, term_coefficient)
            hamiltonian += QubitOperator(yzy_operators, term_coefficient)

    # Return Hamiltonian.
    return hamiltonian
