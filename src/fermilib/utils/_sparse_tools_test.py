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
from scipy.sparse import csc_matrix
import unittest

from fermilib.ops import FermionOperator
from fermilib.transforms import jordan_wigner, get_sparse_operator
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


if __name__ == '__main__':
    unittest.main()
