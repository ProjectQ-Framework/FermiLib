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

"""Module to compute the second order Trotter error for jellium."""
from __future__ import absolute_import
from future.utils import iteritems, itervalues
from math import sqrt

from fermilib.config import *
from fermilib.ops import normal_ordered, FermionOperator
from fermilib.utils._plane_wave_hamiltonian import wigner_seitz_length_scale
from fermilib.utils import jellium_model, Grid

import numpy


def commutator(operator1, operator2):
    """Return the normal-ordered commutator of two fermionic operators.

    Args:
        operator1, operator2: FermionOperators.
    """
    commutator = operator1 * operator2
    commutator -= operator2 * operator1
    commutator = normal_ordered(commutator)
    return commutator


def double_commutator(op1, op2, op3, indices2=None, indices3=None,
                      is_hopping_operator2=None, is_hopping_operator3=None):
    """Return the double commutator [op1, [op2, op3]].

    Assumes the operators are from the jellium Hamiltonian.

    Args:
        op1, op2, op3 (FermionOperators): operators for the commutator.
        indices2, indices3 (set): The indices op2 and op3 act on.
        is_hopping_operator2 (bool): Whether op2 is a hopping operator.
                                     Similarly for is_hopping_operator3.
    """
    if is_hopping_operator2 and is_hopping_operator3:
        indices2 = set(indices2)
        indices3 = set(indices3)
        intersection, = indices2.intersection(indices3)

        indices2.remove(intersection)
        indices3.remove(intersection)

        index2, = indices2
        index3, = indices3
        coeff2 = op2.terms[list(op2.terms)[0]]
        coeff3 = op3.terms[list(op3.terms)[0]]
        commutator23 = (
            FermionOperator(((index2, 1), (index3, 0)), coeff2 * coeff3) +
            FermionOperator(((index3, 1), (index2, 0)), -coeff2 * coeff3))
    else:
        commutator23 = commutator(op2, op3)

    return commutator(op1, commutator23)


def trivially_double_commutes_more_info(indices_alpha=None, indices_beta=None,
                                        indices_alpha_prime=None,
                                        is_hopping_operator_alpha=None,
                                        is_hopping_operator_beta=None,
                                        is_hopping_operator_alpha_prime=None):
    """Determine if [op_a, [op_b, op_a_prime]] is trivially zero.

    Assumes the operators are FermionOperators from the jellium
    Hamiltonian. They are determined by the indices they act on and by
    whether they are hopping operators. a, b, and a_prime are shorthands
    for alpha, beta, and alpha_prime.

    Args:
        indices_alpha (set): The indices term_alpha acts on.
                             Defined similarly for beta and alpha_prime.
        is_hopping_operator_alpha (bool): Whether term_alpha is a
                                          hopping operator. Similarly
                                          for beta, alpha_prime.
    """
    return not ((is_hopping_operator_beta or
                 is_hopping_operator_alpha_prime) and
                (len(indices_beta.intersection(indices_alpha_prime)) == 1) and
                indices_alpha.intersection(
                    indices_beta.union(indices_alpha_prime)))


def trivially_commutes_jellium(term_a, term_b):
    """Determine whether the given terms trivially commute.

    Assumes the terms are single-term FermionOperators from the jellium
    Hamiltonian.

    Args:
        term_a, term_b (FermionOperator): Single-term FermionOperators.
    """
    term_op_a, = term_a.terms.keys()
    term_op_b, = term_b.terms.keys()

    modes_touched_a = [term_op_a[0][0], term_op_a[1][0]]
    modes_touched_b = [term_op_b[0][0], term_op_b[1][0]]

    # If there's no intersection between the modes term_a and term_b act
    # on, the commutator is zero.
    if not (modes_touched_a[0] in modes_touched_b or
            modes_touched_a[1] in modes_touched_b):
        return True

    # In the jellium Hamiltonian, possible number operators take the
    # form a^ a or a^ b^ a b. Number operators always commute trivially.
    term_a_is_number_operator = (term_op_a[0][0] == term_op_a[1][0] or
                                 term_op_a[1][1])
    term_b_is_number_operator = (term_op_b[0][0] == term_op_b[1][0] or
                                 term_op_b[1][1])
    if term_a_is_number_operator and term_b_is_number_operator:
        return True

    # If the first commutator's terms are both hopping, and both create
    # or annihilate the same mode, then the result is zero.
    if not (term_a_is_number_operator or term_b_is_number_operator):
        if (term_op_a[0][0] == term_op_b[0][0] or
                term_op_a[1][0] == term_op_b[1][0]):
            return True

    # If both terms act on the same operators and one is a number operator,
    # they commute.
    if ((term_a_is_number_operator or term_b_is_number_operator) and
            set(modes_touched_a) == set(modes_touched_b)):
        return True

    return False


def trivially_double_commutes_jellium(term_a, term_b, term_c):
    """Check if the double commutator [term_a, [term_b, term_c]] is zero.

    Assumes the terms are single-term FermionOperators from the jellium
    Hamiltonian.

    Args:
        term_a, term_b, term_c: Single-term FermionOperators.

    Notes:
        This function inlines trivially_commutes_jellium for terms b and c.
    """
    # Determine the set of modes each term acts on.
    term_op_b, = term_b.terms.keys()
    term_op_c, = term_c.terms.keys()

    modes_touched_c = [term_op_c[0][0], term_op_c[1][0]]

    # If there's no intersection between the modes term_b and term_c act
    # on, the commutator is trivially zero.
    if not (term_op_b[0][0] in modes_touched_c or
            term_op_b[1][0] in modes_touched_c):
        return True

    # In the jellium Hamiltonian, possible number operators take the
    # form a^ a or a^ b^ a b. Test for this.
    term_b_is_number_operator = (term_op_b[0][0] == term_op_b[1][0] or
                                 term_op_b[1][1])
    term_c_is_number_operator = (term_op_c[0][0] == term_op_c[1][0] or
                                 term_op_c[1][1])

    # Number operators always commute.
    if term_b_is_number_operator and term_c_is_number_operator:
        return True

    # If the first commutator's terms are both hopping, and both create
    # or annihilate the same mode, then the result is zero.
    if not (term_b_is_number_operator or term_c_is_number_operator):
        if (term_op_b[0][0] == term_op_c[0][0] or
                term_op_c[1][0] == term_op_b[1][0]):
            return True

    # The modes term_a acts on are only needed if we reach this stage.
    term_op_a, = term_a.terms.keys()
    modes_touched_b = [term_op_b[0][0], term_op_b[1][0]]
    modes_touched_bc = [term_op_b[0][0], term_op_b[1][0],
                        term_op_c[0][0], term_op_c[1][0]]

    # If the term_a shares no indices with bc, the double commutator is zero.
    if not (term_op_a[0][0] in modes_touched_bc or
            term_op_a[1][0] in modes_touched_bc):
        return True

    # If term_b and term_c are not both number operators and act on the
    # same modes, the commutator is zero.
    if (sum(1 for i in modes_touched_b if i in modes_touched_c) > 1 and
            (term_b_is_number_operator or term_c_is_number_operator)):
        return True

    # Create a list of all the creation and annihilations performed.
    all_changes = term_op_a + term_op_b + term_op_c
    counts = {}
    for operator in all_changes:
        counts[operator[0]] = counts.get(operator[0], 0) + 2 * operator[1] - 1

    # If the final result creates or destroys the same mode twice.
    commutes = max(itervalues(counts)) > 1 or min(itervalues(counts)) < -1

    return commutes


def jellium_error_operator(terms, indices=None, is_hopping_operator=None,
                           series_order=2):
    """Determine the difference between the exact generator of unitary
    evolution and the approximate generator given by Trotter-Suzuki
    to the given order.

    Args:
        terms: a list of FermionOperators in the Hamiltonian
            to be simulated.
        indices: a set of indices the terms act on in the same order as terms.
        is_hopping_operator: a list of whether each term is a hopping operator.
        series_order: the order at which to compute the BCH expansion.
            Only the second order formula is currently implemented
            (corresponding to Equation 9 of the paper).

    Returns:
        The difference between the true and effective generators of time
            evolution for a single Trotter step.

    Notes: follows Equation 9 of Poulin et al.'s work in "The Trotter Step
        Size Required for Accurate Quantum Simulation of Quantum Chemistry".
    """
    if series_order != 2:
        raise NotImplementedError
    more_info = bool(indices)

    error_operator = FermionOperator.zero()
    for beta in range(len(terms)):
        for alpha in range(beta + 1):
            for alpha_prime in range(beta):
                # If both are number operators they commute.
                if more_info and not trivially_double_commutes_more_info(
                        indices[alpha], indices[beta], indices[alpha_prime],
                        is_hopping_operator[alpha], is_hopping_operator[beta],
                        is_hopping_operator[alpha_prime]):
                    # Determine the result of the double commutator.
                    double_com = double_commutator(
                        terms[alpha], terms[beta], terms[alpha_prime],
                        indices[beta], indices[alpha_prime],
                        is_hopping_operator[beta],
                        is_hopping_operator[alpha_prime])
                    if alpha == beta:
                        double_com /= 2.0

                    error_operator += double_com

                elif not trivially_double_commutes_jellium(
                        terms[alpha], terms[beta], terms[alpha_prime]):
                    double_com = double_commutator(
                        terms[alpha], terms[beta], terms[alpha_prime])

                    if alpha == beta:
                        double_com /= 2.0

                    error_operator += double_com

    error_operator /= 12.0
    return error_operator


def jellium_error_bound(terms, indices=None, is_hopping_operator=None,
                        tight=False):
    """
    Numerically upper bound the error in the ground state energy
    for the second order Trotter-Suzuki expansion.

    Args:
        terms: a list of single-term FermionOperators in the Hamiltonian
            to be simulated.
        indices: a set of indices the terms act on in the same order as terms.
        is_hopping_operator: a list of whether each term is a hopping operator.
        tight: whether to use the triangle inequality to give a loose
            upper bound on the error (default) or to calculate the
            norm of the error operator.

    Returns:
        A float upper bound on norm of error in the ground state energy.

    Notes: follows Poulin et al.'s work in "The Trotter Step Size
           Required for Accurate Quantum Simulation of Quantum
           Chemistry". In particular, Equation 16 is used for a loose
           upper bound, and the norm of Equation 9 is calculated for
           a tighter bound using the error operator from
           jellium_error_operator.

           Possible extensions of this function would be to get the
           expectation value of the error operator with the Hartree-Fock
           state or CISD state, which can scalably bound the error in
           the ground state but much more accurately than the triangle
           inequality.
    """
    if tight:
        # Return the 1-norm of the error operator (upper bound on error).
        return sum(numpy.absolute(
            list(jellium_error_operator(
                terms, indices, is_hopping_operator).terms.values())))

    zero = FermionOperator()
    error = 0.0

    for alpha in range(len(terms)):
        term_a = terms[alpha]
        coefficient_a, = term_a.terms.values()
        if coefficient_a:
            error_a = 0.0

            for beta in range(alpha + 1, len(terms)):
                term_b = terms[beta]
                coefficient_b, = term_b.terms.values()
                # If the terms don't commute, add the coefficient.
                if not commutator(term_a, term_b).isclose(zero):
                    error_a += abs(coefficient_b)

            error += 4.0 * abs(coefficient_a) * error_a ** 2

    return error


def ordered_jellium_terms_with_info(
        grid_length, dimension=3, wigner_seitz_radius=10.,
        inverse_filling_fraction=2, spinless=True):
    """Give terms from the jellium Hamiltonian in simulation algorithm order.

    Roughly uses the simulation ordering. Pre-computes term info.

    Args:
        max_grid_length (int): The number of spatial orbitals per
                               dimension.
        dimension (int): The dimension of the system.
        wigner_seitz_radius (float): The radius per particle in Bohr.
        inverse_filling_fraction (int): The ratio of spin-orbitals to
                                        particles.

    Returns:
        A 3-tuple of terms from the jellium Hamiltonian in order of
        simulation, the indices they act on, and whether they are
        hopping operators (both also in the same order).
    """
    zero = FermionOperator.zero()

    n_qubits = grid_length ** dimension
    if not spinless:
        n_qubits *= 2

    # Compute appropriate length scale.
    n_particles = n_qubits // inverse_filling_fraction
    length_scale = wigner_seitz_length_scale(
        wigner_seitz_radius, n_particles, dimension)

    grid = Grid(dimension, grid_length, length_scale)
    hamiltonian = jellium_model(grid, spinless=spinless, plane_wave=False)
    hamiltonian = normal_ordered(hamiltonian)
    hamiltonian.compress()
    ordered_terms = []
    ordered_indices = []
    ordered_is_hopping_operator = []

    # Number of times the two-number term i^ j^ i j appears for fixed i.
    number_number_appearances = [0] * n_qubits
    for i in range(n_qubits):
        for j in range(n_qubits):
            two_number_action = ((i, 1), (j, 1), (i, 0), (j, 0))
            if hamiltonian.terms.get(two_number_action):
                number_number_appearances[i] += 1
                number_number_appearances[j] += 1

    for i in range(n_qubits):
        for j in range(1, n_qubits):
            index_left = min(i, (i + j) % n_qubits)
            index_right = max(i, (i + j) % n_qubits)

            hopping_action1 = ((index_left, 1), (index_right, 0))
            hopping_action2 = ((index_right, 1), (index_left, 0))
            two_number_action = ((index_right, 1), (index_left, 1),
                                 (index_right, 0), (index_left, 0))
            left_number_action = ((index_left, 1), (index_left, 0))
            right_number_action = ((index_right, 1), (index_right, 0))

            hopping_operator = (
                FermionOperator(hopping_action1,
                                hamiltonian.terms.get(hopping_action1, 0.0)) +
                FermionOperator(hopping_action2,
                                hamiltonian.terms.get(hopping_action2, 0.0)))
            hopping_operator.compress()
            hopping_operator /= 2.0

            number_operator = FermionOperator(two_number_action,
                                              hamiltonian.terms.get(
                                                  two_number_action, 0.0))
            if not number_operator.isclose(zero):
                number_operator += (
                    FermionOperator(left_number_action,
                                    hamiltonian.terms.get(
                                        left_number_action, 0.0) /
                                    number_number_appearances[index_left]) +
                    FermionOperator(right_number_action,
                                    hamiltonian.terms.get(
                                        right_number_action, 0.0) /
                                    number_number_appearances[index_right]))
            number_operator.compress()
            number_operator /= 2.0

            if not hopping_operator.isclose(zero):
                ordered_terms.append(hopping_operator)
                ordered_indices.append(set((index_left, index_right)))
                ordered_is_hopping_operator.append(True)
            if not number_operator.isclose(zero):
                ordered_terms.append(number_operator)
                ordered_indices.append(set((index_left, index_right)))
                ordered_is_hopping_operator.append(False)

    return (ordered_terms, ordered_indices, ordered_is_hopping_operator)


def ordered_jellium_terms_no_info(
        grid_length, dimension=3, wigner_seitz_radius=10.,
        inverse_filling_fraction=2, spinless=True):
    """Give terms from the jellium Hamiltonian in simulation algorithm order.

    Gives a fairly arbitrary term ordering.

    Args:
        max_grid_length (int): The number of spatial orbitals per
                               dimension.
        dimension (int): The dimension of the system.
        wigner_seitz_radius (float): The radius per particle in Bohr.
        inverse_filling_fraction (int): The ratio of spin-orbitals to
                                        particles.

    Returns:
        A list of terms from the jellium Hamiltonian in.
    """
    n_qubits = grid_length ** dimension
    n_particles = n_qubits // inverse_filling_fraction
    length_scale = wigner_seitz_length_scale(
        wigner_seitz_radius, n_particles, dimension)

    grid = Grid(dimension, grid_length, length_scale)
    hamiltonian = jellium_model(grid, spinless=spinless, plane_wave=False)
    hamiltonian = normal_ordered(hamiltonian)
    hamiltonian.compress()

    terms = []

    for operators, coefficient in iteritems(hamiltonian.terms):
        terms += [FermionOperator(operators, coefficient)]

    return terms
