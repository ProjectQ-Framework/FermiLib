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

"""BKSF transform on fermionic operators."""
#from __future__ import absolute_import

import numpy as np
from fermilib.ops import InteractionOperator
from projectq.ops import QubitOperator
import networkx as nx


def bksf(operator):
    """ Find the bksf pauli-representation of InteractionOperator. 
        bksf pauli-representation of general FermionOperator is 
        not possible in bksf. So the InteractionOperator given as input must
        be Hermitian.
    Returns:
        transformed_operator: An instance of the QubitOperator class.

    """
    if isinstance(operator, InteractionOperator):
        return bksf_interaction_op(operator)
    else:
        TypeError("operator must be an InteractionOperator.")
  

def bksf_interaction_op(iop, n_qubits=None):
    """Output InteractionOperator as QubitOperator class under BKSF transform.

    Input: Interaction operator to be transformed.
    Returns:
        qubit_operator: An instance of the QubitOperator class.
    """
    from fermilib.utils import count_qubits
    if n_qubits is None:
        n_qubits = count_qubits(iop)
    if n_qubits < count_qubits(iop):
        n_qubits = count_qubits(iop)

    # Initialize qubit operator as constant.
    qubit_operator = QubitOperator((), iop.constant)
    edge_matrix = bksf_edge_matrix(iop)
    # Loop through all indices.
    for p in range(n_qubits):
        for q in range(n_qubits):

            # Handle one-body terms.
            coefficient = complex(iop[p, q])
            if coefficient and p >= q:
                qubit_operator += coefficient * bksf_one_body(edge_matrix, p, q)

            # Keep looping for the two-body terms.
            for r in range(n_qubits):
                for s in range(n_qubits):
                    coefficient = complex(iop[p, q, r, s])

                    # Skip zero terms.
                    if (not coefficient) or (p == q) or (r == s):
                        continue

                    # Identify and skip one of the complex conjugates.
                    if [p, q, r, s] != [s, r, q, p]:
                        if len(set([p, q, r, s])) == 4:
                            if min(r, s) < min(p, q):
                                continue
                        elif p != r and q < p:
                                continue

                    # Handle the two-body terms.
                    transformed_term = bksf_two_body(edge_matrix,p, q, r, s)
                    transformed_term *= coefficient
                    qubit_operator += transformed_term
    return qubit_operator

def bksf_edge_matrix(iop,n_qubits=None):
    """Edge matrix contains the information about the edges between 
        vertices. Edge matrix is required to build the operators in bksf
        model.
        Input: It takes as input interaction operator.
    Returns:
        A square numpy array containing information about the edges present in
        the model
    """
    from fermilib.utils import count_qubits
    if n_qubits is None:
        n_qubits = count_qubits(iop)
    if n_qubits < count_qubits(iop):
        n_qubits = count_qubits(iop)
    edge_matrix=1j*np.zeros((n_qubits,n_qubits))
    # Loop through all indices.
    for p in range(n_qubits):
        for q in range(n_qubits):

            
            # Handle one-body terms.
            coefficient = complex(iop[p, q])
            if coefficient and p >= q:
                edge_matrix[p,q]=bool(complex(iop[p,q]))
                

            # Keep looping for the two-body terms.
            for r in range(n_qubits):
                for s in range(n_qubits):
                    coefficient2 = complex(iop[p, q, r, s])

                    # Skip zero terms.
                    if (not coefficient2) or (p == q) or (r == s):
                        continue

                    # Identify and skip one of the complex conjugates.
                    if [p, q, r, s] != [s, r, q, p]:
                        if len(set([p, q, r, s])) == 4:
                            if min(r, s) < min(p, q):
                                continue
                        elif p != r and q < p:
                                continue
                            
                    if (p == q) or (r == s):
                        continue
            
                    # Handle case of four unique indices.
                    elif len(set([p, q, r, s])) == 4:
                        
                        if coefficient2 and p >= q:
                            edge_matrix[p,q]=bool(complex(iop[p,q,r,s]))
                            a,b = sorted([r,s])
                            edge_matrix[b,a]=bool(complex(iop[p,q,r,s]))
                
                    # Handle case of three unique indices.
                    elif len(set([p, q, r, s])) == 3:
                        
                        # Identify equal tensor factors.
                        if p == r:
                            a, b = sorted([q, s])
                            c = p                            
                            edge_matrix[b,a]=bool(complex(iop[p,q,r,s]))
                        elif p == s:
                            a, b = sorted([q, r])
                            c = p
                            edge_matrix[b,a]=bool(complex(iop[p,q,r,s]))
                        elif q == r:
                            a, b = sorted([p, s])
                            c = q
                            edge_matrix[b,a]=bool(complex(iop[p,q,r,s]))
                        elif q == s:
                            a, b = sorted([p, r])
                            c = q
                            edge_matrix[b,a]=bool(complex(iop[p,q,r,s]))
                        
                    # Handle case of two unique indices.
                    elif len(set([p, q, r, s])) == 2:                        
                        if p == s:
                            a,b=sorted([p,q])
                            edge_matrix[b,a]=bool(complex(iop[p,q,r,s]))
                            
                        else:
                            a,b=sorted([p,q])
                            edge_matrix[b,a]=bool(complex(iop[p,q,r,s]))
            
    return edge_matrix.transpose()
                    
                    
    
def bksf_edge_operator_b(edge_matrix,i):
    """Calculate the edge operator B_i. 
    The definitions used here are consistent with 
     	arXiv:quant-ph/0003137
    Input: It takes as input the edge matrix.
    Returns:
       An instance of qubitoperator
    
    """
    B_i=QubitOperator()
    edge_matrix_indices=np.array(np.nonzero(np.triu(edge_matrix)-np.diag(np.diag(edge_matrix))))    
    #n_qubits=np.size(edge_matrix_indices[0][:])
    qubit_position_matrix=np.array(np.where(edge_matrix_indices==i))
    qubit_position=qubit_position_matrix[1][:]
    qubit_position=np.sort(qubit_position)
    operator=tuple()
    for d1 in qubit_position:
        operator+= ((int(d1),'Z'),)   
    B_i += QubitOperator(operator,1)
    
    return B_i


def bksf_edge_operator_aij(edge_matrix,i,j):
    """Calculate the edge operator A_ij.
    The definitions used here are consistent with 
     	arXiv:quant-ph/0003137
    Input: It takes as input the edge matrix.
    Returns:
       An instance of qubitoperator
    """
    a_ij=QubitOperator()
    edge_matrix_indices=np.array(np.nonzero(np.triu(edge_matrix)-np.diag(np.diag(edge_matrix))))    
    #n_qubits=np.size(edge_matrix_indices[0,:])
    operator=tuple()
    position_ij=-1
    qubit_position_i=np.array(np.where(edge_matrix_indices==i))
    for d1 in range(np.size(edge_matrix_indices[0,:])):
        if set((i,j))==set(edge_matrix_indices[:,d1]):
            position_ij=d1
    operator+=((int(position_ij),'X'),)
    #a_ij+=QubitOperator((position_ij,'X'),1)
    
    
    for d2 in range(np.size(qubit_position_i[0,:])):
        if edge_matrix_indices[int(not(qubit_position_i[0,d2]))][qubit_position_i[1,d2]]<j:
            #a_ij+=QubitOperator((qubit_position_i[1][d2],'Z'),1)
            operator+=((int(qubit_position_i[1,d2]),'Z'),)
    qubit_position_j=np.array(np.where(edge_matrix_indices==j))
    for d3 in range(np.size(qubit_position_j[0,:])):
        if edge_matrix_indices[int(not(qubit_position_j[0,d3]))][qubit_position_j[1,d3]]<i:
            #a_ij+=QubitOperator((qubit_position_j[1][d3],'Z'),1)
            operator+=((int(qubit_position_j[1,d3]),'Z'),)
    a_ij+=QubitOperator(operator,1)
    if j<i:
        a_ij=-1*a_ij
    return a_ij


def bksf_one_body(edge_matrix,p, q):
    """Map the term a^\dagger_p a_q + a^\dagger_q a_p to QubitOperator.
    The definitions for various operators will be presented in a paper soon
    """
    # Handle off-diagonal terms.
    qubit_operator = QubitOperator()
    if p != q:
        a, b = sorted([p, q])
        B_a=bksf_edge_operator_b(edge_matrix,a)
        B_b=bksf_edge_operator_b(edge_matrix,b)
        A_ab=bksf_edge_operator_aij(edge_matrix,a,b)
        qubit_operator+=(-1j/2)*(A_ab*B_b+B_a*A_ab)
        
    # Handle diagonal terms.
    else:
        B_p=bksf_edge_operator_b(edge_matrix,p)
        qubit_operator+=(QubitOperator((),1)-B_p)/2
        
    return qubit_operator


def bksf_two_body(edge_matrix, p, q, r, s):
    """Map the term a^\dagger_p a^\dagger_q a_r a_s + h.c. to QubitOperator.
    The definitions for various operators will be covered in a paper soon.
    """
    # Initialize qubit operator.
    qubit_operator = QubitOperator()

    # Return zero terms.
    if (p == q) or (r == s):
        return qubit_operator

    # Handle case of four unique indices.
    elif len(set([p, q, r, s])) == 4:
        B_p=bksf_edge_operator_b(edge_matrix,p)
        B_q=bksf_edge_operator_b(edge_matrix,q)
        B_r=bksf_edge_operator_b(edge_matrix,r)
        B_s=bksf_edge_operator_b(edge_matrix,s)
        A_pq=bksf_edge_operator_aij(edge_matrix,p,q)
        A_rs=bksf_edge_operator_aij(edge_matrix,r,s)
        qubit_operator+=(1/8)*A_pq*A_rs*(-1*QubitOperator(())-B_p*B_q \
                         +B_p*B_r+B_p*B_s+B_q*B_r+B_q*B_s-B_r*B_s+ \
                         B_p*B_q*B_r*B_s)
        return qubit_operator

    # Handle case of three unique indices.
    elif len(set([p, q, r, s])) == 3:

        # Identify equal tensor factors.
        if p == r:
            B_p=bksf_edge_operator_b(edge_matrix,p)
            B_q=bksf_edge_operator_b(edge_matrix,q)
            B_s=bksf_edge_operator_b(edge_matrix,s)
            A_qs=bksf_edge_operator_aij(edge_matrix,q,s)
            qubit_operator+=(1j/2)*(A_qs*B_s+B_q*A_qs)*(QubitOperator(())-B_p)/2
            
            
        elif p == s:
            B_p=bksf_edge_operator_b(edge_matrix,p)
            B_q=bksf_edge_operator_b(edge_matrix,q)
            B_r=bksf_edge_operator_b(edge_matrix,r)
            A_qr=bksf_edge_operator_aij(edge_matrix,q,r)
            qubit_operator+=(-1j/2)*(A_qr*B_r+B_q*A_qr)*(QubitOperator(())-B_p)/2
            
        elif q == r:
            B_p=bksf_edge_operator_b(edge_matrix,p)
            B_q=bksf_edge_operator_b(edge_matrix,q)
            B_s=bksf_edge_operator_b(edge_matrix,s)
            A_ps=bksf_edge_operator_aij(edge_matrix,p,s)
            qubit_operator+=(-1j/2)*(A_ps*B_s+B_p*A_ps)*(QubitOperator(())-B_q)/2            
            
        elif q == s:
            B_p=bksf_edge_operator_b(edge_matrix,p)
            B_q=bksf_edge_operator_b(edge_matrix,q)
            B_r=bksf_edge_operator_b(edge_matrix,r)
            A_pr=bksf_edge_operator_aij(edge_matrix,p,r)
            qubit_operator+=(1j/2)*(A_pr*B_r+B_p*A_pr)*(QubitOperator(())-B_q)/2
            
    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:

        # Get coefficient.
        if p == s:
            B_p=bksf_edge_operator_b(edge_matrix,p)
            B_q=bksf_edge_operator_b(edge_matrix,q)
            qubit_operator+=(QubitOperator((),1)-B_p)*(QubitOperator((),1)-B_q)/4
            
        else:
            B_p=bksf_edge_operator_b(edge_matrix,p)
            B_q=bksf_edge_operator_b(edge_matrix,q)
            qubit_operator+=-1*(QubitOperator((),1)-B_p)*(QubitOperator((),1)-B_q)/4
                    
    return qubit_operator


def bksf_vacuum(edge_matrix):
    """Use the stabilizers to find the vacuum state in BKSF.
    """
    # Initialize qubit operator.
    edge_matrix_indices=np.array(np.nonzero(np.triu(edge_matrix)-np.diag(np.diag(edge_matrix))))    
    g=nx.Graph()
    g.add_edges_from(tuple(edge_matrix_indices.transpose()))
    stabs=np.array(nx.cycle_basis(g))
     
    #A=np.identity(2**np.size(edge_matrix_indices,1))
    #op=np.zeros(2**np.size(edge_matrix_indices,1),2**np.size(edge_matrix_indices,1))
    #import pdb
    #pdb.set_trace()
    op=QubitOperator(())
    for stab in stabs: 
        
        A=QubitOperator(())
        stab=np.array(stab)
        for i in range(np.size(stab)):
            if i==(np.size(stab)-1):                 
                A=(1j)*A*bksf_edge_operator_aij(edge_matrix,stab[i],stab[0])
                #print('A'+str(stab[i])+str(stab[0]))
            else:                
                A=(1j)*A*bksf_edge_operator_aij(edge_matrix,stab[i],stab[i+1])
                #print('A'+str(stab[i])+str(stab[i+1]))
        op=op*(QubitOperator(())+A)/np.sqrt(2)
        
    return op

def bksf_number_operator(iop, qubit_number=None):
    """Find the qubit operator for the number operator in BKSF representation"""
    n_qubit=iop.n_qubits
    op=QubitOperator()
    edge_matrix=bksf_edge_matrix(iop)
    if qubit_number==None:
        for i in range(n_qubit):
            op+=(QubitOperator(())-bksf_edge_operator_b(edge_matrix,i))/2
            
    else:
        op+=(QubitOperator(())-bksf_edge_operator_b(edge_matrix,qubit_number))/2
        
    return op
        
    
def bksf_generate_fermions(edge_matrix,i,j):
    """The QubitOperator for generating fermions in BKSF representation"""
    op=(-1j/2)*(bksf_edge_operator_aij(edge_matrix,i,j)* \
        bksf_edge_operator_b(edge_matrix,j)-\
        bksf_edge_operator_b(edge_matrix,i)*\
        bksf_edge_operator_aij(edge_matrix,i,j))
    return op

        
    
