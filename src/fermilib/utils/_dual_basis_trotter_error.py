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

"""Module to compute Trotter errors in the plane-wave dual basis."""
from __future__ import absolute_import
from future.utils import iteritems, itervalues
from math import sqrt

import numpy

from fermilib.config import *
from fermilib.ops import FermionOperator, normal_ordered
from fermilib.utils import Grid, jellium_model
from fermilib.utils._operator_utils import commutator, count_qubits
from fermilib.utils._plane_wave_hamiltonian import wigner_seitz_length_scale


def double_commutator(op1, op2, op3, indices2=None, indices3=None,
                      is_hopping_operator2=None, is_hopping_operator3=None):
    """Return the double commutator [op1, [op2, op3]].

    Assumes the operators are from the dual basis Hamiltonian.

    Args:
        op1, op2, op3 (FermionOperators): operators for the commutator.
        indices2, indices3 (set): The indices op2 and op3 act on.
        is_hopping_operator2 (bool): Whether op2 is a hopping operator.
        is_hopping_operator3 (bool): Whether op3 is a hopping operator.

    Returns:
        The double commutator of the given operators.
    """
    if is_hopping_operator2 and is_hopping_operator3:
        indices2 = set(indices2)
        indices3 = set(indices3)
        # Determine which indices both op2 and op3 act on.
        try:
            intersection, = indices2.intersection(indices3)
        except ValueError:
            return FermionOperator.zero()

        # Remove the intersection from the set of indices, since it will get
        # cancelled out in the final result.
        indices2.remove(intersection)
        indices3.remove(intersection)

        # Find the indices of the final output hopping operator.
        index2, = indices2
        index3, = indices3
        coeff2 = op2.terms[list(op2.terms)[0]]
        coeff3 = op3.terms[list(op3.terms)[0]]
        commutator23 = (
            FermionOperator(((index2, 1), (index3, 0)), coeff2 * coeff3) +
            FermionOperator(((index3, 1), (index2, 0)), -coeff2 * coeff3))
    else:
        commutator23 = normal_ordered(commutator(op2, op3))

    return normal_ordered(commutator(op1, commutator23))


def trivially_double_commutes_dual_basis_using_term_info(
        indices_alpha=None, indices_beta=None, indices_alpha_prime=None,
        is_hopping_operator_alpha=None, is_hopping_operator_beta=None,
        is_hopping_operator_alpha_prime=None, jellium_only=False):
    """Return whether [op_a, [op_b, op_a_prime]] is trivially zero.

    Assumes all the operators are FermionOperators from the dual basis
    Hamiltonian, broken into the form i^j^ i j + c_i*(i^ i) + c_j*(j^ j)
    or i^ j + j^ i, where i and j are modes and c is a constant. For the
    full dual basis Hamiltonian, i^ i and j^ j can have distinct
    coefficients c_i and c_j: for jellium they are necessarily the same.
    If this is the case, jellium_only should be set to True.

    The operators are determined by the indices they act on and by
    whether they are hopping operators (i^ j + j^ i) or number operators
    (i^ j^ i j + c_i*(i^ i) + c_j*(j^ j)). a, b, and a_prime are
    shorthands for alpha, beta, and alpha_prime.

    Args:
        indices_alpha (set): The indices term_alpha acts on.
        indices_beta (set): The indices term_beta acts on.
        indices_alpha_prime (set): The indices term_alpha_prime acts on.
        is_hopping_operator_alpha (bool): Whether term_alpha is a
                                          hopping operator.
        is_hopping_operator_beta (bool): Whether term_beta is a
                                         hopping operator.
        is_hopping_operator_alpha_prime (bool): Whether term_alpha_prime
                                                is a hopping operator.
        jellium_only (bool): Whether the terms are only from the jellium
                             Hamiltonian, i.e. if c_i = c for all number
                             operators i^ i or if it depends on i.

    Returns:
        Whether or not the double commutator is trivially zero.
    """
    # If operator_beta and operator_alpha_prime (in the inner commutator)
    # are number operators, they commute trivially.
    if not (is_hopping_operator_beta or is_hopping_operator_alpha_prime):
        return True

    # The operators in the jellium Hamiltonian (provided they are of the
    # form i^ i + j^ j or i^ j^ i j + c*(i^ i + j^ j), and not both
    # hopping operators) commute if they act on the same modes or if
    # there is no intersection.
    if (jellium_only and (not is_hopping_operator_alpha_prime or
                          not is_hopping_operator_beta) and
            len(indices_beta.intersection(indices_alpha_prime)) != 1):
        return True

    # If the modes operator_alpha acts on are disjoint with the modes
    # operator_beta and operator_alpha_prime act on, they commute.
    if not indices_alpha.intersection(indices_beta.union(indices_alpha_prime)):
        return True

    return False


def trivially_commutes_dual_basis(term_a, term_b):
    """Determine whether the given terms trivially commute.

    Assumes the terms are single-term FermionOperators from the
    plane-wave dual basis Hamiltonian.

    Args:
        term_a, term_b (FermionOperator): Single-term FermionOperators.

    Returns:
        Whether or not the commutator is trivially zero.
    """
    modes_acted_on_by_term_a, = term_a.terms.keys()
    modes_acted_on_by_term_b, = term_b.terms.keys()

    modes_touched_a = [modes_acted_on_by_term_a[0][0],
                       modes_acted_on_by_term_a[1][0]]
    modes_touched_b = [modes_acted_on_by_term_b[0][0],
                       modes_acted_on_by_term_b[1][0]]

    # If there's no intersection between the modes term_a and term_b act
    # on, the commutator is zero.
    if not (modes_touched_a[0] in modes_touched_b or
            modes_touched_a[1] in modes_touched_b):
        return True

    # In the dual basis, possible number operators take the form
    # a^ a or a^ b^ a b. Number operators always commute trivially.
    term_a_is_number_operator = (
        modes_acted_on_by_term_a[0][0] == modes_acted_on_by_term_a[1][0] or
        modes_acted_on_by_term_a[1][1])
    term_b_is_number_operator = (
        modes_acted_on_by_term_b[0][0] == modes_acted_on_by_term_b[1][0] or
        modes_acted_on_by_term_b[1][1])
    if term_a_is_number_operator and term_b_is_number_operator:
        return True

    # If the first commutator's terms are both hopping, and both create
    # or annihilate the same mode, then the result is zero.
    if not (term_a_is_number_operator or term_b_is_number_operator):
        if (modes_acted_on_by_term_a[0][0] == modes_acted_on_by_term_b[0][0] or
                modes_acted_on_by_term_a[1][0] ==
                modes_acted_on_by_term_b[1][0]):
            return True

    # If both terms act on the same operators and are not both hopping
    # operators, then they commute.
    if ((term_a_is_number_operator or term_b_is_number_operator) and
            set(modes_touched_a) == set(modes_touched_b)):
        return True

    return False


def trivially_double_commutes_dual_basis(term_a, term_b, term_c):
    """Check if the double commutator [term_a, [term_b, term_c]] is zero.

    Assumes the terms are single-term FermionOperators from the
    plane-wave dual basis Hamiltonian.

    Args:
        term_a, term_b, term_c: Single-term FermionOperators.

    Notes:
        This function inlines trivially_commutes_dual_basis for terms b
        and c.

    Returns:
        Whether or not the double commutator is trivially zero.
    """
    # Determine the set of modes each term acts on.
    modes_acted_on_by_term_b, = term_b.terms.keys()
    modes_acted_on_by_term_c, = term_c.terms.keys()

    modes_touched_c = [modes_acted_on_by_term_c[0][0],
                       modes_acted_on_by_term_c[1][0]]

    # If there's no intersection between the modes term_b and term_c act
    # on, the commutator is trivially zero.
    if not (modes_acted_on_by_term_b[0][0] in modes_touched_c or
            modes_acted_on_by_term_b[1][0] in modes_touched_c):
        return True

    # In the dual_basis Hamiltonian, possible number operators take the
    # form a^ a or a^ b^ a b. Check for this.
    term_b_is_number_operator = (
        modes_acted_on_by_term_b[0][0] == modes_acted_on_by_term_b[1][0] or
        modes_acted_on_by_term_b[1][1])
    term_c_is_number_operator = (
        modes_acted_on_by_term_c[0][0] == modes_acted_on_by_term_c[1][0] or
        modes_acted_on_by_term_c[1][1])

    # Number operators always commute.
    if term_b_is_number_operator and term_c_is_number_operator:
        return True

    # If the first commutator's terms are both hopping, and both create
    # or annihilate the same mode, then the result is zero.
    if not (term_b_is_number_operator or term_c_is_number_operator):
        if (modes_acted_on_by_term_b[0][0] == modes_acted_on_by_term_c[0][0] or
                modes_acted_on_by_term_b[1][0] ==
                modes_acted_on_by_term_c[1][0]):
            return True

    # The modes term_a acts on are only needed if we reach this stage.
    modes_acted_on_by_term_a, = term_a.terms.keys()
    modes_touched_b = [modes_acted_on_by_term_b[0][0],
                       modes_acted_on_by_term_b[1][0]]
    modes_touched_bc = [
        modes_acted_on_by_term_b[0][0], modes_acted_on_by_term_b[1][0],
        modes_acted_on_by_term_c[0][0], modes_acted_on_by_term_c[1][0]]

    # If the term_a shares no indices with bc, the double commutator is zero.
    if not (modes_acted_on_by_term_a[0][0] in modes_touched_bc or
            modes_acted_on_by_term_a[1][0] in modes_touched_bc):
        return True

    # If term_b and term_c are not both number operators and act on the
    # same modes, the commutator is zero.
    if (sum(1 for i in modes_touched_b if i in modes_touched_c) > 1 and
            (term_b_is_number_operator or term_c_is_number_operator)):
        return True

    # Create a list of all the creation and annihilations performed.
    all_changes = (modes_acted_on_by_term_a + modes_acted_on_by_term_b +
                   modes_acted_on_by_term_c)
    counts = {}
    for operator in all_changes:
        counts[operator[0]] = counts.get(operator[0], 0) + 2 * operator[1] - 1

    # If the final result creates or destroys the same mode twice.
    commutes = max(itervalues(counts)) > 1 or min(itervalues(counts)) < -1

    return commutes


def dual_basis_error_operator(terms, indices=None, is_hopping_operator=None,
                              jellium_only=False):
    """Determine the difference between the exact generator of unitary
    evolution and the approximate generator given by the second-order
    Trotter-Suzuki expansion.

    Args:
        terms: a list of FermionOperators in the Hamiltonian in the
               order in which they will be simulated.
        indices: a set of indices the terms act on in the same order as terms.
        is_hopping_operator: a list of whether each term is a hopping operator.
        jellium_only: Whether the terms are from the jellium Hamiltonian only,
                      rather than the full dual basis Hamiltonian (i.e. whether
                      c_i = c for all number operators i^ i, or whether they
                      depend on i as is possible in the general case).

    Returns:
        The difference between the true and effective generators of time
            evolution for a single Trotter step.

    Notes: follows Equation 9 of Poulin et al.'s work in "The Trotter Step
        Size Required for Accurate Quantum Simulation of Quantum Chemistry".
    """
    more_info = bool(indices)

    error_operator = FermionOperator.zero()
    for beta in range(len(terms)):
        for alpha in range(beta + 1):
            for alpha_prime in range(beta):
                # If we have pre-computed info on indices, use it to determine
                # trivial double commutation.
                if more_info:
                    if (not
                        trivially_double_commutes_dual_basis_using_term_info(
                            indices[alpha], indices[beta],
                            indices[alpha_prime], is_hopping_operator[alpha],
                            is_hopping_operator[beta],
                            is_hopping_operator[alpha_prime], jellium_only)):
                        # Determine the result of the double commutator.
                        double_com = double_commutator(
                            terms[alpha], terms[beta], terms[alpha_prime],
                            indices[beta], indices[alpha_prime],
                            is_hopping_operator[beta],
                            is_hopping_operator[alpha_prime])
                        if alpha == beta:
                            double_com /= 2.0

                        error_operator += double_com

                # If we don't have more info, check for trivial double
                # commutation using the terms directly.
                elif not trivially_double_commutes_dual_basis(
                        terms[alpha], terms[beta], terms[alpha_prime]):
                    double_com = double_commutator(
                        terms[alpha], terms[beta], terms[alpha_prime])

                    if alpha == beta:
                        double_com /= 2.0

                    error_operator += double_com

    error_operator /= 12.0
    return error_operator


def dual_basis_error_bound(terms, indices=None, is_hopping_operator=None,
                           jellium_only=False):
    """Numerically upper bound the error in the ground state energy
    for the second-order Trotter-Suzuki expansion.

    Args:
        terms: a list of single-term FermionOperators in the Hamiltonian
            to be simulated.
        indices: a set of indices the terms act on in the same order as terms.
        is_hopping_operator: a list of whether each term is a hopping operator.
        jellium_only: Whether the terms are from the jellium Hamiltonian only,
                      rather than the full dual basis Hamiltonian (i.e. whether
                      c_i = c for all number operators i^ i, or whether they
                      depend on i as is possible in the general case).

    Returns:
        A float upper bound on norm of error in the ground state energy.

    Notes:
        Follows Equation 9 of Poulin et al.'s work in "The Trotter Step
        Size Required for Accurate Quantum Simulation of Quantum
        Chemistry" to calculate the error operator.
    """
    # Return the 1-norm of the error operator (upper bound on error).
    return numpy.sum(numpy.absolute(list(dual_basis_error_operator(
        terms, indices, is_hopping_operator, jellium_only).terms.values())))


def ordered_dual_basis_terms_grouped_by_type_with_info(dual_basis_hamiltonian):
    """Give terms from the dual basis Hamiltonian in simulated order.

    Roughly uses the simulation ordering, grouping terms into hopping
    (i^ j + j^ i) and number (i^j^ i j + c_i i^ i + c_j j^ j) operators.
    Pre-computes term information (indices each operator acts on, as
    well as whether each operator is a hopping operator.

    Args:
        dual_basis_hamiltonian (FermionOperator): The Hamiltonian.

    Returns:
        A 3-tuple of terms from the plane-wave dual basis Hamiltonian in
        order of simulation, the indices they act on, and whether they
        are hopping operators (both also in the same order).
    """
    zero = FermionOperator.zero()
    hamiltonian = dual_basis_hamiltonian

    n_qubits = count_qubits(hamiltonian)

    ordered_terms = []
    ordered_indices = []
    ordered_is_hopping_operator = []

    # Number of times the two-number term i^ j^ i j appears for given i
    # (stored in the ith position of the list).
    two_number_operator_appearances = [0] * n_qubits

    for i in range(n_qubits):
        for j in range(n_qubits):
            two_number_action = ((i, 1), (j, 1), (i, 0), (j, 0))
            if hamiltonian.terms.get(two_number_action):
                # Increment the number of times we've found two-number
                # operators containing i and j.
                two_number_operator_appearances[i] += 1
                two_number_operator_appearances[j] += 1

    # Iterate over the different possible first qubit indices.
    for i in range(n_qubits):
        # Iterate over possible offsets. Each stage of the algorithm
        # simulates a different offset.
        for j in range(1, n_qubits):
            # The index_left is the low index, index_right is high.
            index_left = min(i, (i + j) % n_qubits)
            index_right = max(i, (i + j) % n_qubits)

            # Operators of the hopping term l^ r + r^ l.
            hopping_action1 = ((index_left, 1), (index_right, 0))
            hopping_action2 = ((index_right, 1), (index_left, 0))

            # Operators of the two-number term r^ l^ r l (after normal-
            # ordering).
            two_number_action = ((index_right, 1), (index_left, 1),
                                 (index_right, 0), (index_left, 0))

            # Single-number terms l^ l and r^ r.
            left_number_action = ((index_left, 1), (index_left, 0))
            right_number_action = ((index_right, 1), (index_right, 0))

            # Calculate the hopping operator in the Hamiltonian.
            hopping_operator = (
                FermionOperator(hopping_action1,
                                hamiltonian.terms.get(hopping_action1, 0.0)) +
                FermionOperator(hopping_action2,
                                hamiltonian.terms.get(hopping_action2, 0.0)))
            hopping_operator.compress()
            # Divide by two to avoid double-counting the operator.
            hopping_operator /= 2.0

            # Calculate the two-number operator in the Hamiltonian.
            number_operator = FermionOperator(two_number_action,
                                              hamiltonian.terms.get(
                                                  two_number_action, 0.0))
            # If it's zero, we have nothing to simulate at this stage;
            # otherwise, group l^ l and r^ r with it.
            if not number_operator.isclose(zero):
                number_operator += (
                    FermionOperator(
                        left_number_action,
                        hamiltonian.terms.get(left_number_action, 0.0) /
                        two_number_operator_appearances[index_left]) +
                    FermionOperator(
                        right_number_action,
                        hamiltonian.terms.get(right_number_action, 0.0) /
                        two_number_operator_appearances[index_right]))
            number_operator.compress()
            # Divide by two to avoid double-counting.
            number_operator /= 2.0

            # If the overall hopping operator isn't close to zero, append it.
            # Include the indices it acts on and that it's a hopping operator.
            if not hopping_operator.isclose(zero):
                ordered_terms.append(hopping_operator)
                ordered_indices.append(set((index_left, index_right)))
                ordered_is_hopping_operator.append(True)

            # If the overall number operator isn't close to zero, append it.
            # Include the indices it acts on and that it's a number operator.
            if not number_operator.isclose(zero):
                ordered_terms.append(number_operator)
                ordered_indices.append(set((index_left, index_right)))
                ordered_is_hopping_operator.append(False)

    return (ordered_terms, ordered_indices, ordered_is_hopping_operator)


def ordered_dual_basis_terms_no_info(dual_basis_hamiltonian):
    """Give terms from the dual basis Hamiltonian in simulated order.

    Orders the terms by dictionary output.

    Args:
        dual_basis_hamiltonian (FermionOperator): The Hamiltonian.

    Returns:
        A list of terms from the dual basis Hamiltonian in simulated order.
    """
    n_qubits = count_qubits(dual_basis_hamiltonian)
    terms = []

    for operators, coefficient in iteritems(dual_basis_hamiltonian.terms):
        terms += [FermionOperator(operators, coefficient)]

    return terms


def dual_basis_jellium_hamiltonian(
    grid_length, dimension=3, wigner_seitz_radius=10., n_particles=None,
    spinless=True):
    """Return the jellium Hamiltonian with the given parameters.

    Args:
        grid_length (int): The number of spatial orbitals per dimension.
        dimension (int): The dimension of the system.
        wigner_seitz_radius (float): The radius per particle in Bohr.
        n_particles (int): The number of particles in the system.
                           Defaults to half filling if not specified.
    """
    n_qubits = grid_length ** dimension
    if not spinless:
        n_qubits *= 2

    if n_particles is None:
        # Default to half filling fraction.
        n_particles = n_qubits // 2

    if not (0 <= n_particles <= n_qubits):
        raise ValueError('n_particles must be between 0 and the number of'
                         ' spin-orbitals.')

    # Compute appropriate length scale.
    length_scale = wigner_seitz_length_scale(
        wigner_seitz_radius, n_particles, dimension)

    grid = Grid(dimension, grid_length, length_scale)
    hamiltonian = jellium_model(grid, spinless=spinless, plane_wave=False)
    hamiltonian = normal_ordered(hamiltonian)
    hamiltonian.compress()
    return hamiltonian
