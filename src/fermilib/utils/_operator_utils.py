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

"""This module provides generic tools for classes in ops/"""
from __future__ import absolute_import

import numpy

from fermilib.ops import *

from projectq.ops import QubitOperator


def eigenspectrum(operator):
    """Compute the eigenspectrum of an operator.

    WARNING: This function has cubic runtime in dimension of
        Hilbert space operator, which might be exponential.

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            InteractionTensor, or InteractionRDM.

    Returns:
        eigenspectrum: dense numpy array of floats giving eigenspectrum.
    """
    from fermilib.transforms import get_sparse_operator
    from fermilib.utils import sparse_eigenspectrum
    sparse_operator = get_sparse_operator(operator)
    eigenspectrum = sparse_eigenspectrum(sparse_operator)
    return eigenspectrum


def count_qubits(operator):
    """Compute the minimum number of qubits on which operator acts.

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            InteractionTensor, or InteractionRDM.

    Returns:
        n_qubits (int): The minimum number of qubits on which operator acts.

    Raises:
       TypeError: Operator of invalid type.
    """
    # Handle FermionOperator.
    if isinstance(operator, FermionOperator):
        n_qubits = 0
        for term in operator.terms:
            for ladder_operator in term:
                if ladder_operator[0] + 1 > n_qubits:
                    n_qubits = ladder_operator[0] + 1
        return n_qubits

    # Handle QubitOperator.
    elif isinstance(operator, QubitOperator):
        n_qubits = 0
        for term in operator.terms:
            if term:
                if term[-1][0] + 1 > n_qubits:
                    n_qubits = term[-1][0] + 1
        return n_qubits

    # Handle InteractionOperator, InteractionRDM, InteractionTensor.
    elif isinstance(operator, (InteractionOperator,
                               InteractionRDM,
                               InteractionTensor)):
        return operator.n_qubits

    # Raise for other classes.
    else:
        raise TypeError('Operator of invalid type.')


def is_identity(operator):
    """Check whether QubitOperator of FermionOperator is identity.

    Args:
        operator: QubitOperator or FermionOperator.

    Raises:
        TypeError: Operator of invalid type.
    """
    if isinstance(operator, (QubitOperator, FermionOperator)):
        return list(operator.terms) == [()]
    raise TypeError('Operator of invalid type.')


def commutator(operator_a, operator_b):
    """Compute the commutator of two QubitOperators or FermionOperators."""
    if (isinstance(operator_a, (QubitOperator, FermionOperator)) and
            isinstance(operator_b, (QubitOperator, FermionOperator))):
        return operator_a * operator_b - operator_b * operator_a
    raise TypeError('Operator of invalid type.')
