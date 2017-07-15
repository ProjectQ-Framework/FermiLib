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

"""Tests for jellium_trotter_error.py."""
import unittest

from fermilib.ops import FermionOperator

from fermilib.utils._jellium_trotter_error import *
from fermilib.utils import jellium_model, Grid, wigner_seitz_length_scale


class CommutatorTest(unittest.TestCase):
    def test_commutes_identity(self):
        com = commutator(FermionOperator.identity(),
                         FermionOperator('2^ 3', 2.3))
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_commutes_no_intersection(self):
        com = commutator(FermionOperator('2^ 3'), FermionOperator('4^ 5^ 3'))
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_commutes_number_operators(self):
        com = commutator(FermionOperator('4^ 3^ 4 3'), FermionOperator('2^ 2'))
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_commutator_hopping_operators(self):
        com = commutator(3 * FermionOperator('1^ 2'), FermionOperator('2^ 3'))
        self.assertTrue(com.isclose(FermionOperator('1^ 3', 3)))

    def test_commutator_hopping_with_single_number(self):
        com = commutator(FermionOperator('1^ 2', 1j), FermionOperator('1^ 1'))
        self.assertTrue(com.isclose(-FermionOperator('1^ 2') * 1j))

    def test_commutator_hopping_with_double_number_one_intersection(self):
        com = commutator(FermionOperator('1^ 3'), FermionOperator('3^ 2^ 3 2'))
        self.assertTrue(com.isclose(-FermionOperator('2^ 1^ 3 2')))

    def test_commutator_hopping_with_double_number_two_intersections(self):
        com = commutator(FermionOperator('2^ 3'), FermionOperator('3^ 2^ 3 2'))
        self.assertTrue(com.isclose(FermionOperator.zero()))


class DoubleCommutatorTest(unittest.TestCase):
    def test_double_commutator_no_intersection_with_union_of_second_two(self):
        com = double_commutator(FermionOperator('4^ 3^ 6 5'),
                                FermionOperator('2^ 1 0'),
                                FermionOperator('0^'))
        self.assertTrue(com.isclose(FermionOperator.zero()))

    def test_double_commutator_more_info_not_hopping(self):
        com = double_commutator(
            FermionOperator('3^ 2'),
            FermionOperator('2^ 3') + FermionOperator('3^ 2'),
            FermionOperator('4^ 2^ 4 2'), indices2=set([2, 3]),
            indices3=set([2, 4]), is_hopping_operator2=True,
            is_hopping_operator3=False)
        self.assertTrue(com.isclose(FermionOperator('4^ 2^ 4 2') -
                                    FermionOperator('4^ 3^ 4 3')))

    def test_double_commtator_more_info_both_hopping(self):
        com = double_commutator(
            FermionOperator('4^ 3^ 4 3'),
            FermionOperator('1^ 2', 2.1) + FermionOperator('2^ 1', 2.1),
            FermionOperator('1^ 3', -1.3) + FermionOperator('3^ 1', -1.3),
            indices2=set([1, 2]), indices3=set([1, 3]),
            is_hopping_operator2=True, is_hopping_operator3=True)
        self.assertTrue(com.isclose(FermionOperator('4^ 3^ 4 2', 2.73) +
                                    FermionOperator('4^ 2^ 4 3', 2.73)))


class TriviallyDoubleCommutesMoreInfoTest(unittest.TestCase):
    def test_number_operators_trivially_commute(self):
        self.assertTrue(trivially_double_commutes_more_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([2, 3]),
            is_hopping_operator_alpha=False, is_hopping_operator_beta=False,
            is_hopping_operator_alpha_prime=False))

    def test_left_hopping_operator_no_trivial_commutation(self):
        self.assertFalse(trivially_double_commutes_more_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([2, 3]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False))

    def test_right_hopping_operator_no_trivial_commutation(self):
        self.assertFalse(trivially_double_commutes_more_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([2, 3]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=False,
            is_hopping_operator_alpha_prime=True))

    def test_alpha_is_hopping_operator_others_number_trivial_commutation(self):
        self.assertTrue(trivially_double_commutes_more_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([2, 3]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=False,
            is_hopping_operator_alpha_prime=False))

    def test_no_intersection_in_first_commutator_trivially_commutes(self):
        self.assertTrue(trivially_double_commutes_more_info(
            indices_alpha=set([1, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([1, 2]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False))

    def test_double_intersection_in_first_commutator_trivially_commutes(self):
        self.assertTrue(trivially_double_commutes_more_info(
            indices_alpha=set([3, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([4, 3]),
            is_hopping_operator_alpha=True, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False))

    def test_single_intersection_in_first_commutator_nontrivial(self):
        self.assertFalse(trivially_double_commutes_more_info(
            indices_alpha=set([3, 2]), indices_beta=set([3, 4]),
            indices_alpha_prime=set([4, 5]),
            is_hopping_operator_alpha=False, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False))

    def test_no_intersection_between_first_and_other_terms_is_trivial(self):
        self.assertTrue(trivially_double_commutes_more_info(
            indices_alpha=set([3, 2]), indices_beta=set([1, 4]),
            indices_alpha_prime=set([4, 5]),
            is_hopping_operator_alpha=False, is_hopping_operator_beta=True,
            is_hopping_operator_alpha_prime=False))


class TriviallyCommutesJelliumTest(unittest.TestCase):
    def test_trivially_commutes_no_intersection(self):
        self.assertTrue(trivially_commutes_jellium(
            FermionOperator('3^ 2^ 3 2'), FermionOperator('4^ 1')))

    def test_no_trivial_commute_with_intersection(self):
        self.assertFalse(trivially_commutes_jellium(
            FermionOperator('2^ 1'), FermionOperator('5^ 2^ 5 2')))

    def test_trivially_commutes_both_single_number_operators(self):
        self.assertTrue(trivially_commutes_jellium(
            FermionOperator('3^ 3'), FermionOperator('3^ 3')))

    def test_trivially_commutes_nonintersecting_single_number_operators(self):
        self.assertTrue(trivially_commutes_jellium(
            FermionOperator('2^ 2'), FermionOperator('3^ 3')))

    def test_trivially_commutes_both_double_number_operators(self):
        self.assertTrue(trivially_commutes_jellium(
            FermionOperator('3^ 2^ 3 2'), FermionOperator('3^ 1^ 3 1')))

    def test_trivially_commutes_one_double_number_operators(self):
        self.assertTrue(trivially_commutes_jellium(
            FermionOperator('3^ 2^ 3 2'), FermionOperator('3^ 3')))

    def test_no_trivial_commute_right_hopping_operator(self):
        self.assertFalse(trivially_commutes_jellium(
            FermionOperator('3^ 1^ 3 1'), FermionOperator('3^ 2')))

    def test_no_trivial_commute_left_hopping_operator(self):
        self.assertFalse(trivially_commutes_jellium(
            FermionOperator('3^ 2'), FermionOperator('3^ 3')))

    def test_trivially_commutes_both_hopping_create_same_mode(self):
        self.assertTrue(trivially_commutes_jellium(
            FermionOperator('3^ 2'), FermionOperator('3^ 1')))

    def test_trivially_commutes_both_hopping_annihilate_same_mode(self):
        self.assertTrue(trivially_commutes_jellium(
            FermionOperator('4^ 1'), FermionOperator('3^ 1')))

    def test_trivially_commutes_both_hopping_and_number_on_same_modes(self):
        self.assertTrue(trivially_commutes_jellium(
            FermionOperator('4^ 1'), FermionOperator('4^ 1^ 4 1')))


class TriviallyDoubleCommutesJelliumTest(unittest.TestCase):
    def test_trivially_double_commutes_no_intersection(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('3^ 4'),
            FermionOperator('3^ 2^ 3 2'), FermionOperator('4^ 1')))

    def test_no_trivial_double_commute_with_intersection(self):
        self.assertFalse(trivially_double_commutes_jellium(
            FermionOperator('4^ 2'),
            FermionOperator('2^ 1'), FermionOperator('5^ 2^ 5 2')))

    def test_trivially_double_commutes_both_single_number_operators(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 3'), FermionOperator('3^ 3')))

    def test_trivially_double_commutes_nonintersecting_single_number_ops(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('3^ 2'),
            FermionOperator('2^ 2'), FermionOperator('3^ 3')))

    def test_trivially_double_commutes_both_double_number_operators(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 2^ 3 2'), FermionOperator('3^ 1^ 3 1')))

    def test_trivially_double_commutes_one_double_number_operators(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 2^ 3 2'), FermionOperator('3^ 3')))

    def test_no_trivial_double_commute_right_hopping_operator(self):
        self.assertFalse(trivially_double_commutes_jellium(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 1^ 3 1'), FermionOperator('3^ 2')))

    def test_no_trivial_double_commute_left_hopping_operator(self):
        self.assertFalse(trivially_double_commutes_jellium(
            FermionOperator('4^ 3'),
            FermionOperator('3^ 2'), FermionOperator('3^ 3')))

    def test_trivially_double_commutes_both_hopping_create_same_mode(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('3^ 3'),
            FermionOperator('3^ 2'), FermionOperator('3^ 1')))

    def test_trivially_double_commutes_both_hopping_annihilate_same_mode(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('1^ 1'),
            FermionOperator('4^ 1'), FermionOperator('3^ 1')))

    def test_trivially_double_commutes_hopping_and_number_on_same_modes(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('4^ 3'),
            FermionOperator('4^ 1'), FermionOperator('4^ 1^ 4 1')))

    def test_trivially_double_commutes_no_intersection_a_with_bc(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('5^ 2'),
            FermionOperator('3^ 1'), FermionOperator('4^ 1^ 4 1')))

    def test_trivially_double_commutes_double_create_in_a_and_b(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('5^ 2'),
            FermionOperator('3^ 1'), FermionOperator('4^ 1^ 4 1')))

    def test_trivially_double_commutes_double_annihilate_in_a_and_c(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('5^ 2'),
            FermionOperator('3^ 1'), FermionOperator('4^ 1^ 4 1')))

    def test_no_trivial_double_commute_double_annihilate_with_create(self):
        self.assertFalse(trivially_double_commutes_jellium(
            FermionOperator('5^ 2'),
            FermionOperator('2^ 1'), FermionOperator('4^ 2')))

    def test_trivially_double_commutes_excess_create(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('5^ 2'),
            FermionOperator('5^ 5'), FermionOperator('5^ 1')))

    def test_trivially_double_commutes_excess_annihilate(self):
        self.assertTrue(trivially_double_commutes_jellium(
            FermionOperator('5^ 2'),
            FermionOperator('3^ 2'), FermionOperator('2^ 2')))


class ErrorOperatorTest(unittest.TestCase):
    def test_error_operator(self):
        FO = FermionOperator

        terms = []
        for i in range(4):
            terms.append(FO(((i, 1), (i, 0)), 0.018505508252))
            terms.append(FO(((i, 1), ((i + 1) % 4, 0)), -0.0123370055014))
            terms.append(FO(((i, 1), ((i + 2) % 4, 0)), 0.00616850275068))
            terms.append(FO(((i, 1), ((i + 3) % 4, 0)), -0.0123370055014))
            terms.append(normal_ordered(FO(((i, 1), ((i + 1) % 4, 1),
                                            (i, 0), ((i + 1) % 4, 0)),
                                           3.18309886184)))
            if i // 2:
                terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.2816920329)))

        self.assertAlmostEqual(
            jellium_error_operator(terms).terms[((3, 1), (2, 1), (1, 1),
                                                 (2, 0), (1, 0), (0, 0))],
            -0.562500000003)


class ErrorBoundTest(unittest.TestCase):
    def setUp(self):
        FO = FermionOperator

        self.terms = []
        for i in range(4):
            self.terms.append(FO(((i, 1), (i, 0)), 0.018505508252))
            self.terms.append(FO(((i, 1), ((i + 1) % 4, 0)), -0.0123370055014))
            self.terms.append(FO(((i, 1), ((i + 2) % 4, 0)), 0.00616850275068))
            self.terms.append(FO(((i, 1), ((i + 3) % 4, 0)), -0.0123370055014))
            self.terms.append(normal_ordered(FO(((i, 1), ((i + 1) % 4, 1),
                                                 (i, 0), ((i + 1) % 4, 0)),
                                                3.18309886184)))
            if i // 2:
                self.terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.2816920329)))

    def test_error_bound_tight(self):
        self.assertAlmostEqual(jellium_error_bound(self.terms, tight=True),
                               6.92941899358)

    def test_error_bound_loose(self):
        self.assertAlmostEqual(jellium_error_bound(self.terms), 787.023868666)


class OrderedJelliumTermsMoreInfoTest(unittest.TestCase):
    def test_sum_of_ordered_terms_equals_full_hamiltonian(self):
        grid_length = 4
        dimension = 2
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2

        n_qubits = grid_length ** dimension
        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction
        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, dimension)

        terms = ordered_jellium_terms_with_info(grid_length, dimension)[0]
        terms_total = sum(terms, FermionOperator.zero())

        grid = Grid(dimension, grid_length, length_scale)
        hamiltonian = jellium_model(grid, spinless=True, plane_wave=False)
        hamiltonian = normal_ordered(hamiltonian)
        self.assertTrue(terms_total.isclose(hamiltonian))

    def test_correct_indices_terms_with_info(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2

        n_qubits = grid_length ** dimension
        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction
        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, dimension)

        terms, indices, is_hopping = ordered_jellium_terms_with_info(
            grid_length, dimension)

        for i in range(len(terms)):
            term = list(terms[i].terms)
            term_indices = set()
            for single_term in term:
                term_indices = term_indices.union(
                    [single_term[j][0] for j in range(len(single_term))])
            self.assertEqual(term_indices, indices[i])

    def test_is_hopping_operator_terms_with_info(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2

        n_qubits = grid_length ** dimension
        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction
        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, dimension)

        terms, indices, is_hopping = ordered_jellium_terms_with_info(
            grid_length, dimension)

        for i in range(len(terms)):
            single_term = list(terms[i].terms)[0]
            is_hopping_term = not (single_term[1][1] or
                                   single_term[0][0] == single_term[1][0])
            self.assertEqual(is_hopping_term, is_hopping[i])


class OrderedJelliumTermsNoInfoTest(unittest.TestCase):
    def test_all_terms_in_jellium_hamiltonian(self):
        grid_length = 4
        dimension = 1
        terms = ordered_jellium_terms_no_info(grid_length, dimension)
        FO = FermionOperator

        expected_terms = []
        for i in range(grid_length ** dimension):
            expected_terms.append(FO(((i, 1), (i, 0)),
                                     0.018505508252))
            expected_terms.append(FO(((i, 1), ((i + 1) % 4, 0)),
                                     -0.0123370055014))
            expected_terms.append(FO(((i, 1), ((i + 2) % 4, 0)),
                                     0.00616850275068))
            expected_terms.append(FO(((i, 1), ((i + 3) % 4, 0)),
                                     -0.0123370055014))
            expected_terms.append(normal_ordered(
                FO(((i, 1), ((i + 1) % 4, 1), (i, 0), ((i + 1) % 4, 0)),
                   3.18309886184)))
            if i // 2:
                expected_terms.append(normal_ordered(
                    FO(((i, 1), ((i + 2) % 4, 1), (i, 0), ((i + 2) % 4, 0)),
                       22.2816920329)))

        for term in terms:
            found_in_other = False
            for term2 in expected_terms:
                if term.isclose(term2, rel_tol=1e-8):
                    self.assertFalse(found_in_other)
                    found_in_other = True
            self.assertTrue(found_in_other, msg=str(term))
        for term in expected_terms:
            found_in_other = False
            for term2 in terms:
                if term.isclose(term2, rel_tol=1e-8):
                    self.assertFalse(found_in_other)
                    found_in_other = True
            self.assertTrue(found_in_other, msg=str(term))

    def test_sum_of_ordered_terms_equals_full_hamiltonian(self):
        grid_length = 4
        dimension = 1
        wigner_seitz_radius = 10.0
        inverse_filling_fraction = 2

        n_qubits = grid_length ** dimension
        # Compute appropriate length scale.
        n_particles = n_qubits // inverse_filling_fraction
        length_scale = wigner_seitz_length_scale(
            wigner_seitz_radius, n_particles, dimension)

        terms = ordered_jellium_terms_no_info(grid_length, dimension)
        terms_total = sum(terms, FermionOperator.zero())

        grid = Grid(dimension, grid_length, length_scale)
        hamiltonian = jellium_model(grid, spinless=True, plane_wave=False)
        hamiltonian = normal_ordered(hamiltonian)
        self.assertTrue(terms_total.isclose(hamiltonian))


if __name__ == '__main__':
    unittest.main()
