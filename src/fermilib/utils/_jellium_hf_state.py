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

"""This module constructs the uniform electron gas' Hartree-Fock state."""
from __future__ import absolute_import

from fermilib.ops import FermionOperator, normal_ordered
from fermilib.utils import jellium_model, inverse_fourier_transform

from scipy.sparse import csr_matrix

import numpy


def hartree_fock_state_jellium(grid, n_electrons, spinless=True,
                               plane_wave=False):
    """Give the Hartree-Fock state of jellium.

    Args:
        grid (Grid): The discretization to use.
        n_electrons (int): Number of electrons in the system.
        spinless (bool): Whether to use the spinless model or not.
        plane_wave (bool): Whether to return the Hartree-Fock state in
                           the plane wave (True) or dual basis (False).

    Notes:
        The jellium model is built up by filling the lowest-energy
        single-particle states in the plane-wave Hamiltonian until
        n_electrons states are filled.
    """
    # Get the jellium Hamiltonian in the plane wave basis.
    hamiltonian = jellium_model(grid, spinless, plane_wave=True)
    hamiltonian = normal_ordered(hamiltonian)
    hamiltonian.compress()

    # Enumerate the single-particle states.
    n_single_particle_states = (grid.length ** grid.dimensions)
    if not spinless:
        n_single_particle_states *= 2

    # Compute the energies for each of the single-particle states.
    single_particle_energies = numpy.zeros(n_single_particle_states,
                                           dtype=float)
    for i in range(n_single_particle_states):
        single_particle_energies[i] = hamiltonian.terms.get(
            ((i, 1), (i, 0)), 0.0)

    # The number of occupied single-particle states is the number of electrons.
    # Those states with the lowest single-particle energies are occupied first.
    occupied_states = single_particle_energies.argsort()[:n_electrons]

    if plane_wave:
        # In the plane wave basis the HF state is a single determinant.
        hartree_fock_state_index = numpy.sum(2 ** occupied_states)
        hartree_fock_state = csr_matrix(
            ([1.0], ([hartree_fock_state_index], [0])),
            shape=(2 ** n_single_particle_states, 1))

    else:
        # Inverse Fourier transform the creation operators for the state to get
        # to the dual basis state, then use that to get the dual basis state.
        hartree_fock_state_creation_operator = FermionOperator.identity()
        for state in occupied_states[::-1]:
            hartree_fock_state_creation_operator *= (
                FermionOperator(((int(state), 1),)))
        dual_basis_hf_creation_operator = inverse_fourier_transform(
            hartree_fock_state_creation_operator, grid, spinless)

        dual_basis_hf_creation = normal_ordered(
            dual_basis_hf_creation_operator)

        # Initialize the HF state as a sparse matrix.
        hartree_fock_state = csr_matrix(
            ([], ([], [])), shape=(2 ** n_single_particle_states, 1),
            dtype=complex)

        # Populate the elements of the HF state in the dual basis.
        for term in dual_basis_hf_creation.terms:
            index = 0
            for operator in term:
                index += 2 ** operator[0]
            hartree_fock_state[index, 0] = dual_basis_hf_creation.terms[term]

    return hartree_fock_state
