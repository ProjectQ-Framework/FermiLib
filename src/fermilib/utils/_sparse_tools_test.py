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

"""Tests for sparse_tools.py."""
from __future__ import absolute_import

import numpy
import unittest

from scipy.linalg import eigh, norm
from scipy.sparse import csc_matrix

from fermilib.ops import FermionOperator, number_operator
from fermilib.transforms import get_sparse_operator, jordan_wigner
from fermilib.utils import Grid, jellium_model
from fermilib.utils._sparse_tools import *


class SparseOperatorTest(unittest.TestCase):

    def test_kronecker_operators(self):

        self.assertAlmostEqual(
            0, numpy.amax(numpy.absolute(
                kronecker_operators(3 * [identity_csc]) -
                kronecker_operators(3 * [pauli_x_csc]) ** 2)))

    def test_qubit_jw_fermion_integration(self):

        # Initialize a random fermionic operator.
        fermion_operator = FermionOperator(((3, 1), (2, 1), (1, 0), (0, 0)),
                                           -4.3)
        fermion_operator += FermionOperator(((3, 1), (1, 0)), 8.17)
        fermion_operator += 3.2 * FermionOperator()

        # Map to qubits and compare matrix versions.
        qubit_operator = jordan_wigner(fermion_operator)
        qubit_sparse = get_sparse_operator(qubit_operator)
        qubit_spectrum = sparse_eigenspectrum(qubit_sparse)
        fermion_sparse = jordan_wigner_sparse(fermion_operator)
        fermion_spectrum = sparse_eigenspectrum(fermion_sparse)
        self.assertAlmostEqual(0., numpy.amax(
            numpy.absolute(fermion_spectrum - qubit_spectrum)))

    def test_jw_sparse_index(self):
        """Test the indexing scheme for selecting specific particle numbers"""
        expected = [1, 2]
        calculated_indices = jw_number_indices(1, 2)
        self.assertEqual(expected, calculated_indices)

        expected = [3]
        calculated_indices = jw_number_indices(2, 2)
        self.assertEqual(expected, calculated_indices)

    def test_jw_restrict_operator(self):
        """Test the scheme for restricting JW encoded operators to number"""
        # Make a Hamiltonian that cares mostly about number of electrons
        n_qubits = 6
        target_electrons = 3
        penalty_const = 100.
        number_sparse = jordan_wigner_sparse(number_operator(n_qubits))
        bias_sparse = jordan_wigner_sparse(
            sum([FermionOperator(((i, 1), (i, 0)), 1.0) for i
                 in range(n_qubits)], FermionOperator()))
        hamiltonian_sparse = penalty_const * (
            number_sparse - target_electrons *
            scipy.sparse.identity(2**n_qubits)).dot(
            number_sparse - target_electrons *
            scipy.sparse.identity(2**n_qubits)) + bias_sparse

        restricted_hamiltonian = jw_number_restrict_operator(
            hamiltonian_sparse, target_electrons, n_qubits)
        true_eigvals, _ = eigh(hamiltonian_sparse.A)
        test_eigvals, _ = eigh(restricted_hamiltonian.A)

        self.assertAlmostEqual(norm(true_eigvals[:20] - test_eigvals[:20]),
                               0.0)

    def test_jw_restrict_operator_hopping_to_1_particle(self):
        hop = FermionOperator('3^ 1') + FermionOperator('1^ 3')
        hop_sparse = jordan_wigner_sparse(hop, n_qubits=4)
        hop_restrict = jw_number_restrict_operator(hop_sparse, 1, n_qubits=4)
        expected = csc_matrix(([1, 1], ([0, 2], [2, 0])), shape=(4, 4))

        self.assertTrue(numpy.allclose(hop_restrict.A, expected.A))

    def test_jw_restrict_operator_interaction_to_1_particle(self):
        interaction = FermionOperator('3^ 2^ 4 1')
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        interaction_restrict = jw_number_restrict_operator(
            interaction_sparse, 1, n_qubits=6)
        expected = csc_matrix(([], ([], [])), shape=(6, 6))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_operator_interaction_to_2_particles(self):
        interaction = (FermionOperator('3^ 2^ 4 1') +
                       FermionOperator('4^ 1^ 3 2'))
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        interaction_restrict = jw_number_restrict_operator(
            interaction_sparse, 2, n_qubits=6)

        dim = 6 * 5 / 2  # shape of new sparse array
        # 3^ 2^ 4 1 maps 2**4 + 2 = 18 to 2**3 + 2**2 = 12 and vice versa;
        # in the 2-particle subspace (1, 4) and (2, 3) are 7th and 9th.
        expected = csc_matrix(([-1, -1], ([7, 9], [9, 7])), shape=(dim, dim))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_operator_hopping_to_1_particle_default_nqubits(self):
        interaction = (FermionOperator('3^ 2^ 4 1') +
                       FermionOperator('4^ 1^ 3 2'))
        interaction_sparse = jordan_wigner_sparse(interaction, n_qubits=6)
        # n_qubits should default to 6
        interaction_restrict = jw_number_restrict_operator(
            interaction_sparse, 2)

        dim = 6 * 5 / 2  # shape of new sparse array
        # 3^ 2^ 4 1 maps 2**4 + 2 = 18 to 2**3 + 2**2 = 12 and vice versa;
        # in the 2-particle subspace (1, 4) and (2, 3) are 7th and 9th.
        expected = csc_matrix(([-1, -1], ([7, 9], [9, 7])), shape=(dim, dim))

        self.assertTrue(numpy.allclose(interaction_restrict.A, expected.A))

    def test_jw_restrict_jellium_ground_state_integration(self):
        n_qubits = 4
        grid = Grid(dimensions=1, length=n_qubits, scale=1.0)
        jellium_hamiltonian = jordan_wigner_sparse(
            jellium_model(grid, spinless=False))

        #  2 * n_qubits because of spin
        number_sparse = jordan_wigner_sparse(number_operator(2 * n_qubits))

        restricted_number = jw_number_restrict_operator(number_sparse, 2)
        restricted_jellium_hamiltonian = jw_number_restrict_operator(
            jellium_hamiltonian, 2)

        energy, ground_state = get_ground_state(restricted_jellium_hamiltonian)

        number_expectation = expectation(restricted_number, ground_state)
        self.assertAlmostEqual(number_expectation, 2)


class JordanWignerSparseTest(unittest.TestCase):

    def test_jw_sparse_0create(self):
        expected = csc_matrix(([1], ([1], [0])), shape=(2, 2))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('0^')).A,
            expected.A))

    def test_jw_sparse_1annihilate(self):
        expected = csc_matrix(([1, 1], ([0, 2], [1, 3])), shape=(4, 4))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('1')).A,
            expected.A))

    def test_jw_sparse_0create_2annihilate(self):
        expected = csc_matrix(([-1j, 1j],
                               ([4, 6], [1, 3])),
                              shape=(8, 8))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('0^ 2', -1j)).A,
            expected.A))

    def test_jw_sparse_0create_3annihilate(self):
        expected = csc_matrix(([-1j, 1j, 1j, -1j],
                               ([8, 10, 12, 14], [1, 3, 5, 7])),
                              shape=(16, 16))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('0^ 3', -1j)).A,
            expected.A))

    def test_jw_sparse_twobody(self):
        expected = csc_matrix(([1, 1], ([6, 14], [5, 13])), shape=(16, 16))
        self.assertTrue(numpy.allclose(
            jordan_wigner_sparse(FermionOperator('2^ 1^ 1 3')).A,
            expected.A))

    def test_qubit_operator_sparse_n_qubits_too_small(self):
        with self.assertRaises(ValueError):
            qubit_operator_sparse(QubitOperator('X3'), 1)

    def test_qubit_operator_sparse_n_qubits_not_specified(self):
        expected = csc_matrix(([1, 1, 1, 1], ([1, 0, 3, 2], [0, 1, 2, 3])),
                              shape=(4, 4))
        self.assertTrue(numpy.allclose(
            qubit_operator_sparse(QubitOperator('X1')).A,
            expected.A))


class GroundStateTest(unittest.TestCase):
    def test_get_ground_state_hermitian(self):
        ground = get_ground_state(get_sparse_operator(
            QubitOperator('Y0 X1') + QubitOperator('Z0 Z1')))
        expected_state = csc_matrix(([1j, 1], ([1, 2], [0, 0])),
                                    shape=(4, 1)).A
        expected_state /= numpy.sqrt(2.0)

        self.assertAlmostEqual(ground[0], -2)
        self.assertAlmostEqual(
            numpy.absolute(expected_state.T.dot(ground[1].A))[0, 0], 1.0)

    def test_get_ground_state_nonhermitian(self):
        with self.assertRaises(ValueError):
            get_ground_state(get_sparse_operator(1j * QubitOperator('X1')))


class ExpectationTest(unittest.TestCase):
    def test_expectation_correct(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = csc_matrix(([1j, 1j], ([1, 3], [0, 0])), shape=(4, 1))
        self.assertAlmostEqual(expectation(operator, vector), 2.0)

    def test_expectation_correct_zero(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = csc_matrix(([1j, -1j, -1j, -1j],
                             ([0, 1, 2, 3], [0, 0, 0, 0])), shape=(4, 1))
        self.assertAlmostEqual(expectation(operator, vector), 0.0)

    def test_expectation_invalid_state_length(self):
        operator = get_sparse_operator(QubitOperator('X0'), n_qubits=2)
        vector = csc_matrix(([1j, -1j, -1j],
                             ([0, 1, 2], [0, 0, 0])), shape=(3, 1))
        with self.assertRaises(ValueError):
            expectation(operator, vector)


class GetGapTest(unittest.TestCase):
    def test_get_gap(self):
        operator = QubitOperator('Y0 X1') + QubitOperator('Z0 Z1')
        self.assertAlmostEqual(get_gap(get_sparse_operator(operator)), 2.0)

    def test_get_gap_nonhermitian_error(self):
        operator = (QubitOperator('X0 Y1', 1 + 1j) +
                    QubitOperator('Z0 Z1', 1j) + QubitOperator((), 2 + 1j))
        with self.assertRaises(ValueError):
            get_gap(get_sparse_operator(operator))
