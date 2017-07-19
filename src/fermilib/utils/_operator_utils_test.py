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

"""Tests for operator_utils."""
from __future__ import absolute_import

import numpy
import os
import unittest

from fermilib.config import *
from fermilib.ops import *
from fermilib.transforms import jordan_wigner, get_interaction_operator
from fermilib.utils import (eigenspectrum, commutator,
                            count_qubits, is_identity)
from fermilib.utils._operator_utils import (commutator, count_qubits,
                                            eigenspectrum, get_file_path,
                                            is_identity, load_operator,
                                            OperatorUtilsError, save_operator)

from projectq.ops import QubitOperator


class OperatorUtilsTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.interaction_operator = get_interaction_operator(
            self.fermion_operator)

    def test_n_qubits_single_fermion_term(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.fermion_term))

    def test_n_qubits_fermion_operator(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.fermion_operator))

    def test_n_qubits_qubit_operator(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.qubit_operator))

    def test_n_qubits_interaction_operator(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.interaction_operator))

    def test_n_qubits_bad_type(self):
        with self.assertRaises(TypeError):
            count_qubits('twelve')

    def test_eigenspectrum(self):
        fermion_eigenspectrum = eigenspectrum(self.fermion_operator)
        qubit_eigenspectrum = eigenspectrum(self.qubit_operator)
        interaction_eigenspectrum = eigenspectrum(self.interaction_operator)
        for i in range(2 ** self.n_qubits):
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   qubit_eigenspectrum[i])
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   interaction_eigenspectrum[i])

    def test_is_identity_unit_fermionoperator(self):
        self.assertTrue(is_identity(FermionOperator(())))

    def test_is_identity_double_of_unit_fermionoperator(self):
        self.assertTrue(is_identity(2. * FermionOperator(())))

    def test_is_identity_unit_qubitoperator(self):
        self.assertTrue(is_identity(QubitOperator(())))

    def test_is_identity_double_of_unit_qubitoperator(self):
        self.assertTrue(is_identity(QubitOperator((), 2.)))

    def test_not_is_identity_single_term_fermionoperator(self):
        self.assertFalse(is_identity(FermionOperator('1^')))

    def test_not_is_identity_single_term_qubitoperator(self):
        self.assertFalse(is_identity(QubitOperator('X1')))

    def test_not_is_identity_zero_fermionoperator(self):
        self.assertFalse(is_identity(FermionOperator()))

    def test_not_is_identity_zero_qubitoperator(self):
        self.assertFalse(is_identity(QubitOperator()))

    def test_is_identity_bad_type(self):
        with self.assertRaises(TypeError):
            is_identity('eleven')


class SaveLoadOperatorTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)

    def test_save_and_load_operators(self):
        file_name = "test_file"

        save_operator(self.fermion_operator, file_name)
        loaded_fermion_operator = load_operator(file_name)
        self.assertEqual(self.fermion_operator.terms,
                         loaded_fermion_operator.terms)
        os.remove(get_file_path(file_name, DATA_DIRECTORY))

        save_operator(self.qubit_operator, file_name)
        loaded_qubit_operator = load_operator(file_name)
        self.assertEqual(self.qubit_operator.terms,
                         loaded_qubit_operator.terms)
        os.remove(get_file_path(file_name, DATA_DIRECTORY))

    def test_save_and_load_operators_errors(self):
        file_name = "test_file"

        with self.assertRaises(OperatorUtilsError):
            save_operator("invalid_operator_type")
        with self.assertRaises(OperatorUtilsError):
            save_operator(self.fermion_operator)

        save_operator(self.fermion_operator, file_name)
        with self.assertRaises(OperatorUtilsError):
            save_operator(self.fermion_operator, file_name)
        os.remove(get_file_path(file_name, DATA_DIRECTORY))

        constant = 100.0
        one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
        two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                self.n_qubits, self.n_qubits), float)
        one_body[1, 1] = 10.0
        two_body[1, 2, 3, 4] = 12.0
        interaction_operator = InteractionOperator(
            constant, one_body, two_body)
        with self.assertRaises(NotImplementedError):
            save_operator(interaction_operator, file_name)

    def test_save_bad_type(self):
        with self.assertRaises(TypeError):
            save_operator('ping', 'somewhere')


class CommutatorTest(unittest.TestCase):

    def setUp(self):
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
            self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)

    def test_commutes_identity(self):
        com = commutator(FermionOperator.identity(),
                         FermionOperator('2^ 3', 2.3))
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_commutes_no_intersection(self):
        com = commutator(FermionOperator('2^ 3'), FermionOperator('4^ 5^ 3'))
        com = normal_ordered(com)
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_commutes_number_operators(self):
        com = commutator(FermionOperator('4^ 3^ 4 3'), FermionOperator('2^ 2'))
        com = normal_ordered(com)
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_commutator_hopping_operators(self):
        com = commutator(3 * FermionOperator('1^ 2'), FermionOperator('2^ 3'))
        com = normal_ordered(com)
        self.assertTrue(com.isclose(FermionOperator('1^ 3', 3)))

    def test_commutator_hopping_with_single_number(self):
        com = commutator(FermionOperator('1^ 2', 1j), FermionOperator('1^ 1'))
        com = normal_ordered(com)
        self.assertTrue(com.isclose(-FermionOperator('1^ 2') * 1j))

    def test_commutator_hopping_with_double_number_one_intersection(self):
        com = commutator(FermionOperator('1^ 3'), FermionOperator('3^ 2^ 3 2'))
        com = normal_ordered(com)
        self.assertTrue(com.isclose(-FermionOperator('2^ 1^ 3 2')))

    def test_commutator_hopping_with_double_number_two_intersections(self):
        com = commutator(FermionOperator('2^ 3'), FermionOperator('3^ 2^ 3 2'))
        com = normal_ordered(com)
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_commutator(self):
        operator_a = FermionOperator('')
        self.assertTrue(FermionOperator().isclose(
            commutator(operator_a, self.fermion_operator)))
        operator_b = QubitOperator('X1 Y2')
        self.assertTrue(commutator(self.qubit_operator, operator_b).isclose(
            self.qubit_operator * operator_b -
            operator_b * self.qubit_operator))

    def test_commutator_operator_b_bad_type_raise_TypeError(self):
        with self.assertRaises(TypeError):
            commutator(1, self.fermion_operator)

    def test_commutator_operator_b_bad_type_raise_TypeError(self):
        with self.assertRaises(TypeError):
            commutator(self.qubit_operator, "hello")

    def test_commutator_not_same_type(self):
        with self.assertRaises(TypeError):
            commutator(self.fermion_operator, self.qubit_operator)
