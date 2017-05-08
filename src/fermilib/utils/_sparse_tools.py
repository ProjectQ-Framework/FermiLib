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

"""This module provides functions to interface with scipy.sparse."""
from __future__ import absolute_import

from functools import reduce
import numpy
import numpy.linalg
import scipy
import scipy.sparse
import scipy.sparse.linalg

from fermilib.config import *

from projectq.ops import QubitOperator


# Make global definitions.
identity_csc = scipy.sparse.identity(2, format='csr', dtype=complex)
pauli_x_csc = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
pauli_y_csc = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
pauli_z_csc = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex)
q_raise_csc = (pauli_x_csc - 1.j * pauli_y_csc) / 2.
q_lower_csc = (pauli_x_csc + 1.j * pauli_y_csc) / 2.
pauli_matrix_map = {'I': identity_csc, 'X': pauli_x_csc,
                    'Y': pauli_y_csc, 'Z': pauli_z_csc}


def wrapped_kronecker(operator_1, operator_2):
    """Return the Kronecker product of two sparse.csc_matrix operators."""
    return scipy.sparse.kron(operator_1, operator_2, 'csc')


def kronecker_operators(*args):
    """Return the Kronecker product of multiple sparse.csc_matrix operators."""
    return reduce(wrapped_kronecker, *args)


def jordan_wigner_ladder_sparse(n_qubits, tensor_factor, ladder_type):
    """Make a matrix representation of a fermion ladder operator.

    Args:
        index: This is a nonzero integer. The integer indicates the tensor
            factor and the sign indicates raising or lowering.
        n_qubits(int): Number qubits in the system Hilbert space.

    Returns:
        The corresponding SparseOperator.
    """
    identities = [scipy.sparse.identity(
        2 ** tensor_factor, dtype=complex, format='csc')]
    parities = (n_qubits - tensor_factor - 1) * [pauli_z_csc]
    if ladder_type:
        operator = kronecker_operators(identities + [q_raise_csc] + parities)
    else:
        operator = kronecker_operators(identities + [q_lower_csc] + parities)
    return operator


def jordan_wigner_sparse(fermion_operator, n_qubits=None):
    """Initialize a SparseOperator from a FermionOperator.

    Args:
        fermion_operator(FermionOperator): instance of the FermionOperator
            class.
        n_qubits(int): Number of qubits.

    Returns:
        The corresponding SparseOperator.
    """
    if n_qubits is None:
        from fermilib.utils import count_qubits
        n_qubits = count_qubits(fermion_operator)

    # Create a list of raising and lowering operators for each orbital.
    jw_operators = []
    for tensor_factor in range(n_qubits):
        jw_operators += [(jordan_wigner_ladder_sparse(n_qubits,
                                                      tensor_factor,
                                                      0),
                          jordan_wigner_ladder_sparse(n_qubits,
                                                      tensor_factor,
                                                      1))]

    # Construct the SparseOperator.
    n_hilbert = 2 ** n_qubits
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]
        sparse_matrix = coefficient * scipy.sparse.identity(
            2 ** n_qubits, dtype=complex, format='csc')
        for ladder_operator in term:
            sparse_matrix = sparse_matrix * jw_operators[
                ladder_operator[0]][ladder_operator[1]]

        if coefficient:
            # Extract triplets from sparse_term.
            sparse_matrix = sparse_matrix.tocoo(copy=False)
            values_list.append(sparse_matrix.data)
            (row, column) = sparse_matrix.nonzero()
            row_list.append(row)
            column_list.append(column)

    values_list = numpy.concatenate(values_list)
    row_list = numpy.concatenate(row_list)
    column_list = numpy.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix((
        values_list, (row_list, column_list)),
        shape=(n_hilbert, n_hilbert)).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def qubit_operator_sparse(qubit_operator, n_qubits):
    """Initialize a SparseOperator from a QubitOperator.

    Args:
        qubit_operator(QubitOperator): instance of the QubitOperator class.
        n_qubits (int): Number of qubits.

    Returns:
        The corresponding SparseOperator.
    """
    if n_qubits is None:
        from fermilib.utils import count_qubits
        n_qubits = count_qubits(fermion_operator)

    # Construct the SparseOperator.
    n_hilbert = 2 ** n_qubits
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]

    # Loop through the terms.
    for qubit_term in qubit_operator.terms:
        tensor_factor = 0
        coefficient = qubit_operator.terms[qubit_term]
        sparse_operators = [coefficient]
        for pauli_operator in qubit_term:

            # Grow space for missing identity operators.
            if pauli_operator[0] > tensor_factor:
                identity_qubits = pauli_operator[0] - tensor_factor
                identity = scipy.sparse.identity(
                    2 ** identity_qubits, dtype=complex, format='csc')
                sparse_operators += [identity]

            # Add actual operator to the list.
            sparse_operators += [pauli_matrix_map[pauli_operator[1]]]
            tensor_factor = pauli_operator[0] + 1

        # Grow space at end of string unless operator acted on final qubit.
        if tensor_factor < n_qubits or not qubit_term:
            identity_qubits = n_qubits - tensor_factor
            identity = scipy.sparse.identity(
                2 ** identity_qubits, dtype=complex, format='csc')
            sparse_operators += [identity]

        # Extract triplets from sparse_term.
        sparse_matrix = kronecker_operators(sparse_operators)
        values_list.append(sparse_matrix.tocoo(copy=False).data)
        (column, row) = sparse_matrix.nonzero()
        column_list.append(column)
        row_list.append(row)

    # Create sparse operator.
    values_list = numpy.concatenate(values_list)
    row_list = numpy.concatenate(row_list)
    column_list = numpy.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix((
        values_list, (row_list, column_list)),
        shape=(n_hilbert, n_hilbert)).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def jw_hartree_fock_state(n_electrons, n_orbitals):
    """Function to product Hartree-Fock state in JW representation."""
    occupied = scipy.sparse.csr_matrix([[0], [1]], dtype=float)
    psi = 1.
    unoccupied = scipy.sparse.csr_matrix([[1], [0]], dtype=float)
    for orbital in range(n_electrons):
        psi = scipy.sparse.kron(psi, occupied, 'csr')
    for orbital in range(n_orbitals - n_electrons):
        psi = scipy.sparse.kron(psi, unoccupied, 'csr')
    return psi


def get_density_matrix(states, probabilities):
    n_qubits = states[0].shape[0]
    density_matrix = scipy.sparse.csc_matrix(
        (n_qubits, n_qubits), dtype=complex)
    for state, probability in zip(states, probabilities):
        density_matrix = density_matrix + probability * state * state.getH()
    return density_matrix


def is_hermitian(sparse_operator):
    """Test if matrix is Hermitian."""
    difference = sparse_operator - sparse_operator.getH()
    if difference.nnz:
        discrepancy = max(map(abs, difference.data))
        if discrepancy > EQ_TOLERANCE:
            return False
    return True


def get_ground_state(sparse_operator):
    """Compute lowest eigenvalue and eigenstate.

    Returns:
        eigenvalue: The lowest eigenvalue, a float.
        eigenstate: The lowest eigenstate in scipy.sparse csc format.
    """
    if is_hermitian(sparse_operator):
        values, vectors = scipy.sparse.linalg.eigsh(
            sparse_operator, 2, which='SA', maxiter=1e7)
    else:
        values, vectors = scipy.sparse.linalg.eigs(
            sparse_operator, 2, which='SA', maxiter=1e7)
    eigenstate = scipy.sparse.csc_matrix(vectors[:, 0])
    eigenvalue = values[0]
    return eigenvalue, eigenstate.getH()


def sparse_eigenspectrum(sparse_operator):
    """Perform a dense diagonalization.

    Returns:
        eigenspectrum: The lowest eigenvalues in a numpy array.
    """
    dense_operator = sparse_operator.todense()
    if is_hermitian(sparse_operator):
        eigenspectrum = numpy.linalg.eigvalsh(dense_operator)
    else:
        eigenspectrum = numpy.linalg.eigvals(dense_operator)
    return eigenspectrum


def expectation(sparse_operator, state):
    """Compute expectation value of operator with a state.

    Args:
        state_vector: scipy.sparse.csc vector representing a pure state,
            or, a scipy.sparse.csc matrix representing a density matrix.

    Returns:
        A real float giving expectation value.

    Raises:
        SparseOperatorError: Input state has invalid format.
    """
    # Handle density matrix.
    if state.shape == sparse_operator.shape:
        product = state * sparse_operator
        expectation = numpy.sum(product.diagonal())

    elif state.shape == (sparse_operator.shape[0], 1):
        # Handle state vector.
        expectation = state.getH() * sparse_operator * state
        expectation = expectation[0, 0]

    else:
        # Handle exception.
        raise SparseOperatorError('Input state has invalid format.')

    # Return.
    return expectation


def get_gap(sparse_operator):
    """Compute gap between lowest eigenvalue and first excited state.

    Returns: A real float giving eigenvalue gap.
    """
    if is_hermitian(sparse_operator):
        values, _ = scipy.sparse.linalg.eigsh(
            sparse_operator, 2, which='SA', maxiter=1e7)
    else:
        values, _ = scipy.sparse.linalg.eigs(
            sparse_operator, 2, which='SA', maxiter=1e7)
    gap = abs(values[1] - values[0])
    return gap
