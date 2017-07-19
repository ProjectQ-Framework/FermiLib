#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""Tests  _bksf_test.py."""
from __future__ import absolute_import
import numpy as np

import unittest

from fermilib.ops import (FermionOperator,
                          hermitian_conjugated,
                          InteractionOperator,
                          normal_ordered,
                          number_operator)
from fermilib.transforms import (get_interaction_operator,
                                 reverse_jordan_wigner)
from fermilib.transforms._jordan_wigner import (jordan_wigner,
                                                jordan_wigner_one_body,
                                                jordan_wigner_two_body)
from _bksf import (bksf_edge_matrix,
                                                bksf,
                                                bksf_edge_operator_aij,
                                                bksf_edge_operator_b,
                                                bksf_generate_fermions,
                                                bksf_vacuum,
                                                bksf_number_operator)
from fermilib.utils import *


from projectq.ops import QubitOperator


class bksfTransformTest(unittest.TestCase):
    def setUp(self):
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = os.path.join(THIS_DIRECTORY, 'data', 'H2_sto-3g_singlet')
        self.molecule = MolecularData(
            geometry, basis, multiplicity, filename=filename)
        self.molecule.load()

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()

        # Get FCI RDM.
        self.fci_rdm = self.molecule.get_molecular_rdm(use_fci=1)
        #pdb.set_trace()
        # Get explicit coefficients.
        self.nuclear_repulsion = self.molecular_hamiltonian.constant
        self.one_body = self.molecular_hamiltonian.one_body_tensor
        self.two_body = self.molecular_hamiltonian.two_body_tensor

        # Get fermion Hamiltonian.
        self.fermion_hamiltonian = normal_ordered(get_fermion_operator(
            self.molecular_hamiltonian))

        # Get qubit Hamiltonian.
        self.qubit_hamiltonian = jordan_wigner(self.fermion_hamiltonian)

        # Get the sparse matrix.
        self.hamiltonian_matrix = get_sparse_operator(
            self.molecular_hamiltonian)


    def test_bad_input(self):
        with self.assertRaises(TypeError):
            bksf(FermionOperator((2,1),1))

    def test_bksf_edgeoperator_Bi(self):
        import _bksf
        edge_matrix=np.triu(np.ones((4,4)))

        correct_operators_b0 = ((0, 'Z'), (1, 'Z'), (2, 'Z'))
        correct_operators_b1 = ((0, 'Z'), (3, 'Z'), (4, 'Z'))
        correct_operators_b2 = ((1, 'Z'), (3, 'Z'), (5, 'Z'))
        correct_operators_b3 = ((2, 'Z'), (4, 'Z'), (5, 'Z'))
        
        qterm_b0 = QubitOperator(correct_operators_b0, 1)
        qterm_b1 = QubitOperator(correct_operators_b1, 1)
        qterm_b2 = QubitOperator(correct_operators_b2, 1)
        qterm_b3 = QubitOperator(correct_operators_b3, 1)
        
        
        self.assertTrue(qterm_b0.isclose(_bksf.bksf_edge_operator_b(edge_matrix,0)))
        self.assertTrue(qterm_b1.isclose(_bksf.bksf_edge_operator_b(edge_matrix,1)))
        self.assertTrue(qterm_b2.isclose(_bksf.bksf_edge_operator_b(edge_matrix,2)))
        self.assertTrue(qterm_b3.isclose(_bksf.bksf_edge_operator_b(edge_matrix,3)))
    


    def test_bksf_edgeoperator_Aij(self):
        import _bksf
        edge_matrix=np.triu(np.ones((4,4)))

        correct_operators_a01 = ((0, 'X'),)
        correct_operators_a02 = ((0, 'Z'), (1, 'X'))
        correct_operators_a03 = ((0, 'Z'), (1, 'Z'), (2, 'X'))
        correct_operators_a12 = ((0, 'Z'), (1, 'Z'), (3, 'X'))
        correct_operators_a13 = ((0, 'Z'), (2, 'Z'), (3, 'Z'),(4,'X'))
        correct_operators_a23 = ((1,'Z'), (2, 'Z'), (3,'Z'),(4, 'Z'), (5, 'X'))
        
        qterm_a01 = QubitOperator(correct_operators_a01, 1)
        qterm_a02 = QubitOperator(correct_operators_a02, 1)
        qterm_a03 = QubitOperator(correct_operators_a03, 1)
        qterm_a12 = QubitOperator(correct_operators_a12, 1)
        qterm_a13 = QubitOperator(correct_operators_a13, 1)
        qterm_a23 = QubitOperator(correct_operators_a23, 1)
        
        
        self.assertTrue(qterm_a01.isclose(_bksf.bksf_edge_operator_aij(edge_matrix,0,1)))
        self.assertTrue(qterm_a02.isclose(_bksf.bksf_edge_operator_aij(edge_matrix,0,2)))
        self.assertTrue(qterm_a03.isclose(_bksf.bksf_edge_operator_aij(edge_matrix,0,3)))
        self.assertTrue(qterm_a12.isclose(_bksf.bksf_edge_operator_aij(edge_matrix,1,2)))                
        self.assertTrue(qterm_a13.isclose(_bksf.bksf_edge_operator_aij(edge_matrix,1,3)))
        self.assertTrue(qterm_a23.isclose(_bksf.bksf_edge_operator_aij(edge_matrix,2,3)))


    def test_bksf_jw_number_operator(self):
        bksf_n = bksf_number_operator(self.molecular_hamiltonian)
        from fermilib.utils import count_qubits
        jw_n=QubitOperator()
        for i in range(count_qubits(self.molecular_hamiltonian)):
            jw_n+=jordan_wigner_one_body(i,i)
        jw_eig_spec= eigenspectrum(jw_n)
        bksf_eig_spec=eigenspectrum(bksf_n)
        """for i in range(np.size(bksf_eig_spec)):
            if i<np.size(jw_eig_spec):
                print("BK    "+str(bksf_eig_spec[i])+"  JW    "+str(jw_eig_spec[i]))
            else:
                print("BK    "+str(bksf_eig_spec[i]))"""
        evensector=0
        for i in range(np.size(jw_eig_spec)):
            if bool(np.size(np.where(jw_eig_spec[i]==bksf_eig_spec))):
                evensector+=1
                
        #print("The number of eigenvalues of the Number Operator in Jordan-Wigner that are found in BKSF are "+str(evensector))
        
    def test_bksf_jw_hamiltonian(self):
        bksf_H = bksf(self.molecular_hamiltonian)
        jw_H = jordan_wigner(self.molecular_hamiltonian)
        bksf_H_eig=eigenspectrum(bksf_H)
        jw_H_eig=eigenspectrum(jw_H)
        bksf_H_eig=bksf_H_eig.round(5)
        jw_H_eig=jw_H_eig.round(5)
        """for i in range(np.size(bksf_H_eig)):
            if i<np.size(jw_H_eig):
                print("BK    "+str(bksf_H_eig[i])+"  JW    "+str(jw_H_eig[i]))
            else:
                print("BK    "+str(bksf_H_eig[i]))"""
        evensector=0
        for i in range(np.size(jw_H_eig)):
            if bool(np.size(np.where(jw_H_eig[i]==bksf_H_eig))):
                evensector+=1
                
        #print("The number of eigenvalues of the Hamiltonian in Jordan-Wigner that are found in BKSF are "+str(evensector))
        




if __name__ == '__main__':
    unittest.main()
