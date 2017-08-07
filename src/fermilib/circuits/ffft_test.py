"""Tests for ffft.py."""
import numpy
import numpy.linalg
import os
import random
import unittest

from fermilib.circuits.ffft import (
    ffft, fswap, fswap_adjacent, fswap_generator, apply_phase,
    fourier_transform_0, swap_adjacent_fermionic_modes, ffft_2d,
    operator_2d_fft_with_reordering)
from fermilib.ops import FermionOperator, normal_ordered
from fermilib.transforms import jordan_wigner
from fermilib.utils import (count_qubits, eigenspectrum, fourier_transform,
                            Grid, jellium_model)

from projectq import MainEngine
from projectq.ops import (QubitOperator, H, X, Y, Z, C, Rx, Ry, Rz, Swap, Ph,
                          Measure, All, TimeEvolution)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def prepare_logical_state(register, n):
    for i in range(int(numpy.log2(n + 1)) + 1):
        if (n >> i) & 1:
            X | register[i]


def prepare_integer_fermion_operator(n):
    res = FermionOperator('')
    for i in range(int(numpy.log2(n + 1)) + 1):
        if (n >> i) & 1:
            res = FermionOperator(((i, 1),)) * res
    return res


def ordered_wavefunction(engine, indices_to_evaluate=None):
    """Return the correctly ordered wave function amplitudes.

    Args:
        engine (projectq.MainEngine): The engine from which to take the
                                      amplitudes.
        indices_to_evaluate (numpy.array): Array of integer indices to
                                           evaluate amplitudes of. If
                                           indices_to_evaluate is not
                                           set, all amplitudes are
                                           evaluated.
    """
    n_qubits = engine._qubit_idx
    if indices_to_evaluate is None:
        indices_to_evaluate = numpy.arange(2 ** n_qubits)
    elif isinstance(indices_to_evaluate, list):
        indices_to_evaluate = numpy.array(indices_to_evaluate)
    indices = numpy.zeros(len(indices_to_evaluate), dtype=int)

    # Get the qubit order dictionary and raw wavefunction from the engine.
    qubit_order_dict, unordered_wavefunction = engine.backend.cheat()

    for bit_number in range(n_qubits):
        # To each logical state for which qubit k is 1, add
        # 2 ** (qubit k's position in the raw wavefunction). This is quickly
        # given by 1 << qubit_order_dict[k].
        indices[(indices_to_evaluate & (1 << bit_number)) != 0] += (
            1 << qubit_order_dict[bit_number])

    # Return the subset of unordered_wavefunction corresponding to the indices.
    return list(map(unordered_wavefunction.__getitem__, indices))


class OrderedWavefunctionTest(unittest.TestCase):
    def setUp(self):
        self.eng1 = MainEngine()
        self.reg1 = self.eng1.allocate_qubit()
        self.eng3 = MainEngine()
        self.reg3 = self.eng3.allocate_qureg(3)

    def tearDown(self):
        All(Measure) | self.reg1
        All(Measure) | self.reg3

    def test_correct_phase_after_reordering_1qubit(self):
        H | self.reg1
        Rz(0.03) | self.reg1
        self.eng1.flush()
        angle0 = numpy.angle(ordered_wavefunction(self.eng1)[0])
        angle1 = numpy.angle(ordered_wavefunction(self.eng1)[1])
        self.assertAlmostEqual(angle1 - angle0, 0.03)

    def test_correct_phase_after_reordering_multiple_qubits(self):
        All(H) | self.reg3
        Rz(0.07) | self.reg3[1]
        self.eng3.flush()
        wavefunction = ordered_wavefunction(self.eng3)
        self.assertAlmostEqual(numpy.angle(wavefunction[1]),
                               numpy.angle(wavefunction[0]))
        self.assertAlmostEqual(numpy.angle(wavefunction[6]),
                               numpy.angle(wavefunction[2]))
        self.assertAlmostEqual(numpy.angle(wavefunction[2]) -
                               numpy.angle(wavefunction[0]),
                               0.07)

    def test_correct_resorting(self):
        """For this circuit:
        000 -> 0
        1 = 100 CZ[0, 2]-> 100 Z[1]-> 100 Swap[1, 2]-> 100 Z[0]-> -100
            Swap[0, 1]-> 010 = -2
        2 = 010 Z[1]-> -010 Swap[1, 2]-> -001 = -4
        3 = 110 Z[1]-> -110 Swap[1, 2]-> -101 -Z[0]-> 101 Swap[0, 1]-> 6
        4 = 001 Swap[1, 2]-> 010 Swap[0, 1]-> 100 = 1
        5 = 101 CZ[0, 2]-> -101 Swap[1, 2]-> -110 -Z[0]-> 110 = 3
        6 = 011 Z[1]-> -011 CZ[1, 2]-> 011 Swap[0, 1] -> 101 = 5
        7 = 111 CZ[1, 2]-> -111 Z[1]-> 111 CZ[1, 2]-> -111 Z[0]-> 111 = 7

        so the correct signs at the output are +, +, -, +, -, +, +, +.
        """
        All(H) | self.reg3
        C(Z) | (self.reg3[0], self.reg3[2])
        Z | self.reg3[1]
        Swap | (self.reg3[1], self.reg3[2])
        C(Z) | (self.reg3[1], self.reg3[2])
        Z | self.reg3[0]
        Swap | (self.reg3[1], self.reg3[0])
        self.eng3.flush()

        expected = (numpy.array([1, 1, -1, 1, -1, 1, 1, 1]) /
                    (2 * numpy.sqrt(2)))

        self.assertTrue(numpy.allclose(
            numpy.array(ordered_wavefunction(self.eng3)),
            expected))

    def test_correct_resorting_selective(self):
        """Same circuit as non-selective test:
        000 -> 0
        1 = 100 CZ[0, 2]-> 100 Z[1]-> 100 Swap[1, 2]-> 100 Z[0]-> -100
            Swap[0, 1]-> 010 = -2
        2 = 010 Z[1]-> -010 Swap[1, 2]-> -001 = -4
        3 = 110 Z[1]-> -110 Swap[1, 2]-> -101 -Z[0]-> 101 Swap[0, 1]-> 6
        4 = 001 Swap[1, 2]-> 010 Swap[0, 1]-> 100 = 1
        5 = 101 CZ[0, 2]-> -101 Swap[1, 2]-> -110 -Z[0]-> 110 = 3
        6 = 011 Z[1]-> -011 CZ[1, 2]-> 011 Swap[0, 1] -> 101 = 5
        7 = 111 CZ[1, 2]-> -111 Z[1]-> 111 CZ[1, 2]-> -111 Z[0]-> 111 = 7

        so the correct signs at the output are +, +, -, +, -, +, +, +.
        """
        All(H) | self.reg3
        C(Z) | (self.reg3[0], self.reg3[2])
        Z | self.reg3[1]
        Swap | (self.reg3[1], self.reg3[2])
        C(Z) | (self.reg3[1], self.reg3[2])
        Z | self.reg3[0]
        Swap | (self.reg3[1], self.reg3[0])
        self.eng3.flush()

        expected = numpy.array([-1, -1]) / (2 * numpy.sqrt(2))

        self.assertTrue(numpy.allclose(
            numpy.array(ordered_wavefunction(self.eng3, [2, 4])),
            expected))


class FSwapTest(unittest.TestCase):
    def setUp(self):
        self.eng = MainEngine()
        self.reg = self.eng.allocate_qureg(3)

    def tearDown(self):
        All(Measure) | self.reg

    def test_fswap_identity(self):
        fswap(self.reg, 0, 1)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([1.0] + [0.0] * 7)
        self.assertTrue(numpy.allclose(numpy.abs(ordered_wvfn),
                                       numpy.abs(expected)))

    def test_fswap_100(self):
        X | self.reg[0]
        fswap(self.reg, 0, 1)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([0.0] * 2 + [1.0] + [0.0] * 5)
        self.assertTrue(numpy.allclose(numpy.abs(ordered_wvfn),
                                       numpy.abs(expected)))

    def test_fswap_010(self):
        X | self.reg[1]
        fswap(self.reg, 0, 1)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([0.0, 1.0] + [0.0] * 6)
        self.assertTrue(numpy.allclose(numpy.abs(ordered_wvfn),
                                       numpy.abs(expected)))

    def test_fswap_001(self):
        X | self.reg[2]
        fswap(self.reg, 0, 1)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([0.0] * 4 + [1.0] + [0.0] * 3)
        self.assertTrue(numpy.allclose(numpy.abs(ordered_wvfn),
                                       numpy.abs(expected)))

    def test_fswap_superposition(self):
        All(H) | self.reg
        fswap(self.reg, 0, 1)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([2. ** -1.5] * 8)
        expected[3] *= -1  # bitstring 110 gets sign-flipped by fermionic swap
        expected[7] *= -1  # bitstring 111 gets sign-flipped by fermionic swap
        self.assertTrue(numpy.allclose(ordered_wvfn, expected * -1j))

    def test_fswap_same_modes(self):
        All(H) | self.reg
        fswap(self.reg, 1, 1)  # expect identity to be applied because of this
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([2. ** -1.5] * 8)
        self.assertTrue(numpy.allclose(ordered_wvfn, expected * -1j))


class FSwapAdjacentTest(unittest.TestCase):
    def setUp(self):
        self.eng = MainEngine()
        self.reg = self.eng.allocate_qureg(3)

    def tearDown(self):
        All(Measure) | self.reg

    def test_fswap_adjacent_identity(self):
        fswap_adjacent(self.reg, 0)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([1.0] + [0.0] * 7)
        self.assertTrue(numpy.allclose(numpy.abs(ordered_wvfn),
                                       numpy.abs(expected)))

    def test_fswap_adjacent_100(self):
        X | self.reg[0]
        fswap_adjacent(self.reg, 0)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([0.0] * 2 + [1.0] + [0.0] * 5)
        self.assertTrue(numpy.allclose(numpy.abs(ordered_wvfn),
                                       numpy.abs(expected)))

    def test_fswap_adjacent_010(self):
        X | self.reg[1]
        fswap_adjacent(self.reg, 0)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([0.0, 1.0] + [0.0] * 6)
        self.assertTrue(numpy.allclose(numpy.abs(ordered_wvfn),
                                       numpy.abs(expected)))

    def test_fswap_adjacent_001(self):
        X | self.reg[2]
        fswap_adjacent(self.reg, 0)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([0.0] * 4 + [1.0] + [0.0] * 3)
        self.assertTrue(numpy.allclose(numpy.abs(ordered_wvfn),
                                       numpy.abs(expected)))

    def test_fswap_adjacent_superposition(self):
        All(H) | self.reg
        fswap_adjacent(self.reg, 0)
        self.eng.flush()

        ordered_wvfn = numpy.array(ordered_wavefunction(self.eng))
        expected = numpy.array([2. ** -1.5] * 8)
        expected[3] *= -1  # bitstring 110 gets sign-flipped by fermionic swap
        expected[7] *= -1  # bitstring 111 gets sign-flipped by fermionic swap
        self.assertTrue(numpy.allclose(ordered_wvfn, expected * -1j))


class ApplyPhaseTest(unittest.TestCase):
    def setUp(self):
        self.eng = MainEngine()
        self.reg = self.eng.allocate_qureg(3)

    def tearDown(self):
        All(Measure) | self.reg

    def test_apply_phase_1qubit(self):
        eng = MainEngine()
        reg = eng.allocate_qubit()
        H | reg
        apply_phase(reg, 0, 0.03)
        eng.flush()
        self.assertAlmostEqual(numpy.angle(ordered_wavefunction(eng)[1]) -
                               numpy.angle(ordered_wavefunction(eng)[0]),
                               0.06)
        Measure | reg

    def test_apply_phase_multiple_qubits(self):
        All(H) | self.reg
        apply_phase(self.reg, 1, 0.07)
        self.eng.flush()
        wavefunction = ordered_wavefunction(self.eng)
        self.assertAlmostEqual(numpy.angle(wavefunction[1]),
                               numpy.angle(wavefunction[0]))
        self.assertAlmostEqual(numpy.angle(wavefunction[6]),
                               numpy.angle(wavefunction[2]))
        self.assertAlmostEqual(numpy.angle(wavefunction[2]) -
                               numpy.angle(wavefunction[0]),
                               0.14)


class FFFTNModeIntegrationTest(unittest.TestCase):

    def test_ffft_0mode_error(self):
        eng = MainEngine()
        reg = eng.allocate_qureg(1)
        with self.assertRaises(ValueError):
            ffft(eng, reg, 0)

    def test_ffft_1mode_error(self):
        eng = MainEngine()
        reg = eng.allocate_qureg(1)
        with self.assertRaises(ValueError):
            ffft(eng, reg, 1)

    def test_ffft_2modes_properly_applied(self):
        eng_ft_0 = MainEngine()
        reg_ft_0 = eng_ft_0.allocate_qureg(2)

        eng_ffft_recursive = MainEngine()
        reg_ffft_recursive = eng_ffft_recursive.allocate_qureg(2)

        ffft(eng_ffft_recursive, reg_ffft_recursive, 2)
        fourier_transform_0(reg_ft_0, 0, 1)
        eng_ffft_recursive.flush()
        eng_ft_0.flush()

        All(Measure) | reg_ffft_recursive
        All(Measure) | reg_ft_0

        self.assertTrue(numpy.allclose(
            ordered_wavefunction(eng_ffft_recursive),
            ordered_wavefunction(eng_ft_0)))

    def test_2mode_ffft_correct_frequencies(self):
        n_qubits = 2
        frequencies_seen = numpy.zeros(n_qubits)

        for qubit in range(n_qubits):
            engine = MainEngine()
            register = engine.allocate_qureg(n_qubits)
            X | register[qubit]
            ffft(engine, register, n_qubits)
            engine.flush()

            wavefunction = ordered_wavefunction(engine)
            nonzero_wavefunction_elmts = []
            for el in wavefunction:
                if abs(el) > 10 ** -5:
                    nonzero_wavefunction_elmts.append(el)
            All(Measure) | register

            phase_factor = (nonzero_wavefunction_elmts[1] /
                            nonzero_wavefunction_elmts[0])
            offset = numpy.angle(nonzero_wavefunction_elmts[0])

            self.assertAlmostEqual(offset, 0.0)
            for i in range(1, len(nonzero_wavefunction_elmts)):
                self.assertAlmostEqual(phase_factor,
                                       (nonzero_wavefunction_elmts[i] /
                                        nonzero_wavefunction_elmts[i - 1]))
            frequencies_seen[qubit] = numpy.angle(phase_factor)

        frequencies_seen = numpy.sort(frequencies_seen)
        expected = numpy.sort(2 * numpy.pi * numpy.arange(n_qubits) / n_qubits)

        self.assertTrue(numpy.allclose(frequencies_seen, expected))

    def test_4mode_ffft_correct_frequencies(self):
        n_qubits = 4
        frequencies_seen = numpy.zeros(n_qubits)
        offset = None

        for qubit in range(n_qubits):
            engine = MainEngine()
            register = engine.allocate_qureg(n_qubits)
            X | register[qubit]
            ffft(engine, register, n_qubits)
            engine.flush()

            wavefunction = ordered_wavefunction(engine)
            nonzero_wavefunction_elmts = []
            for el in wavefunction:
                if abs(el) > 10 ** -5:
                    nonzero_wavefunction_elmts.append(el)
            All(Measure) | register
            engine.flush()

            phase_factor = (nonzero_wavefunction_elmts[1] /
                            nonzero_wavefunction_elmts[0])
            if offset is None:
                offset = numpy.angle(nonzero_wavefunction_elmts[0])
            self.assertAlmostEqual(offset,
                                   numpy.angle(nonzero_wavefunction_elmts[0]))

            for i in range(1, len(nonzero_wavefunction_elmts)):
                self.assertAlmostEqual(phase_factor,
                                       (nonzero_wavefunction_elmts[i] /
                                        nonzero_wavefunction_elmts[i - 1]))
            frequencies_seen[qubit] = numpy.angle(phase_factor)
            if frequencies_seen[qubit] < -(10 ** -5):
                frequencies_seen[qubit] += 2 * numpy.pi

        frequencies_seen = numpy.sort(frequencies_seen)
        expected = numpy.sort(2 * numpy.pi * numpy.arange(n_qubits) / n_qubits)

        self.assertTrue(numpy.allclose(frequencies_seen, expected))

    def test_8mode_ffft_correct_frequencies(self):
        n_qubits = 8
        frequencies_seen = numpy.zeros(n_qubits)
        offset = None

        for qubit in range(n_qubits):
            engine = MainEngine()
            register = engine.allocate_qureg(n_qubits)
            X | register[qubit]
            ffft(engine, register, n_qubits)
            engine.flush()

            wavefunction = ordered_wavefunction(engine)
            nonzero_wavefunction_elmts = []
            for el in wavefunction:
                if abs(el) > 10 ** -5:
                    nonzero_wavefunction_elmts.append(el)

            All(Measure) | register

            phase_factor = (nonzero_wavefunction_elmts[1] /
                            nonzero_wavefunction_elmts[0])
            if offset is None:
                offset = numpy.angle(nonzero_wavefunction_elmts[0])
            self.assertAlmostEqual(offset,
                                   numpy.angle(nonzero_wavefunction_elmts[0]))

            for i in range(1, len(nonzero_wavefunction_elmts)):
                self.assertAlmostEqual(phase_factor,
                                       (nonzero_wavefunction_elmts[i] /
                                        nonzero_wavefunction_elmts[i - 1]),
                                       msg="wvfn = {}".format(
                                           nonzero_wavefunction_elmts))
            frequencies_seen[qubit] = numpy.angle(phase_factor)
            if frequencies_seen[qubit] < -(10 ** -5):
                frequencies_seen[qubit] += 2 * numpy.pi

        frequencies_seen = numpy.sort(frequencies_seen)
        expected = numpy.sort(2 * numpy.pi * numpy.arange(n_qubits) / n_qubits)

        self.assertTrue(numpy.allclose(frequencies_seen, expected))


class SwapAdjacentFermionicModesTest(unittest.TestCase):
    def test_basic_swap(self):
        operator = normal_ordered(FermionOperator('2^ 2 4^ 4'))
        operator_swapped = swap_adjacent_fermionic_modes(operator, 2)

        self.assertTrue(operator_swapped.isclose(
            FermionOperator('4^ 3^ 4 3', -1.0)))


class FFFTPlaneWaveIntegrationTest(unittest.TestCase):
    def test_4mode_ffft_with_external_swaps_all_logical_states(self):
        n_qubits = 4

        grid = Grid(dimensions=1, length=n_qubits, scale=1.0)

        for i in range(2 ** n_qubits):
            eng = MainEngine()
            register = eng.allocate_qureg(n_qubits)
            prepare_logical_state(register, i)

            ffft(eng, register, n_qubits)
            Ph(3 * numpy.pi / 4) | register
            eng.flush()
            wvfn = ordered_wavefunction(eng)
            All(Measure) | register

            fermion_operator = prepare_integer_fermion_operator(i)

            # Reorder the modes for correct input to the FFFT.
            # Swap 0123 to 2301 for fourier_transform. Additionally, the
            # FFFT's ordering is 0213, so connect 0213 -> 0123 -> 2301.
            swap_mode_list = [1] + [1, 0, 2, 1]
            for mode in swap_mode_list:
                fermion_operator = normal_ordered(fermion_operator)
                fermion_operator = swap_adjacent_fermionic_modes(
                    fermion_operator, mode)

            ffft_result = fourier_transform(fermion_operator,
                                            grid, spinless=True)
            ffft_result = normal_ordered(ffft_result)

            swap_mode_list = [1, 0, 2, 1]  # After FFFT, swap 2301 -> 0123
            for mode in swap_mode_list:
                ffft_result = swap_adjacent_fermionic_modes(ffft_result, mode)

            converted_wvfn = numpy.zeros(2 ** n_qubits, dtype=complex)
            for term in ffft_result.terms:
                index = sum(2 ** site[0] for site in term)
                converted_wvfn[index] = ffft_result.terms[term]

            self.assertTrue(numpy.allclose(wvfn, converted_wvfn))

    def test_8mode_ffft_with_external_swaps_on_single_logical_state(self):
        n_qubits = 8
        grid = Grid(dimensions=1, length=n_qubits, scale=1.0)

        eng = MainEngine()
        register = eng.allocate_qureg(n_qubits)

        state_index = 157

        prepare_logical_state(register, state_index)

        ffft(eng, register, n_qubits)
        Ph(3 * numpy.pi / 4) | register
        eng.flush()
        wvfn = ordered_wavefunction(eng)
        All(Measure) | register

        fermion_operator = prepare_integer_fermion_operator(state_index)

        # Swap 01234567 to 45670123 for fourier_transform. Additionally,
        # the FFFT's ordering is 04261537, so swap 04261537 to 01234567,
        # and then 01234567 to 45670123.
        swap_mode_list = ([1, 3, 5, 2, 4, 1, 3, 5] +
                          [3, 2, 4, 1, 3, 5, 0, 2, 4, 6, 1, 3, 5, 2, 4, 3])
        for mode in swap_mode_list:
            fermion_operator = normal_ordered(fermion_operator)
            fermion_operator = swap_adjacent_fermionic_modes(
                fermion_operator, mode)

        ffft_result = fourier_transform(fermion_operator,
                                        grid, spinless=True)
        ffft_result = normal_ordered(ffft_result)

        # After the FFFT, swap 45670123 -> 01234567.
        swap_mode_list = [3, 2, 4, 1, 3, 5, 0, 2, 4, 6, 1, 3, 5, 2, 4, 3]
        for mode in swap_mode_list:
            ffft_result = swap_adjacent_fermionic_modes(ffft_result, mode)

        converted_wvfn = numpy.zeros(2 ** n_qubits, dtype=complex)
        for term in ffft_result.terms:
            index = sum(2 ** site[0] for site in term)
            converted_wvfn[index] = ffft_result.terms[term]

        self.assertTrue(numpy.allclose(wvfn, converted_wvfn))

    def test_4mode_ffft_with_external_swaps_equal_expectation_values(self):
        n_qubits = 4

        grid = Grid(dimensions=1, length=n_qubits, scale=1.0)
        dual_basis = jellium_model(grid, spinless=True, plane_wave=False)
        ffft_result = normal_ordered(dual_basis)

        # Reorder the modes for correct input to the FFFT.
        # Swap 0123 to 2301 for fourier_transform. Additionally, the
        # FFFT's ordering is 0213, so connect 0213 -> 0123 -> 2301.
        swap_mode_list = [1] + [1, 0, 2, 1]
        for mode in swap_mode_list:
            ffft_result = swap_adjacent_fermionic_modes(ffft_result, mode)
            ffft_result = normal_ordered(ffft_result)

        ffft_result = fourier_transform(ffft_result,
                                        grid, spinless=True)
        ffft_result = normal_ordered(ffft_result)

        swap_mode_list = [1, 0, 2, 1]  # After FFFT, swap 2301 -> 0123
        for mode in swap_mode_list:
            ffft_result = swap_adjacent_fermionic_modes(ffft_result, mode)
            ffft_result = normal_ordered(ffft_result)

        jw_dual_basis = jordan_wigner(dual_basis)
        jw_plane_wave = jordan_wigner(ffft_result)

        # Do plane wave and dual basis calculations simultaneously.
        pw_engine = MainEngine()
        pw_wavefunction = pw_engine.allocate_qureg(n_qubits)
        pw_engine.flush()
        db_engine = MainEngine()
        db_wavefunction = db_engine.allocate_qureg(n_qubits)
        db_engine.flush()

        # Choose random state.
        state = numpy.zeros(2 ** n_qubits, dtype=complex)
        for i in range(len(state)):
            state[i] = (random.random() *
                        numpy.exp(1j * 2 * numpy.pi * random.random()))
        state /= numpy.linalg.norm(state)

        # Put randomly chosen state in the registers.
        pw_engine.backend.set_wavefunction(state, pw_wavefunction)
        db_engine.backend.set_wavefunction(state, db_wavefunction)

        prepare_logical_state(pw_wavefunction, i)
        prepare_logical_state(db_wavefunction, i)

        All(H) | [pw_wavefunction[1], pw_wavefunction[3]]
        All(H) | [db_wavefunction[1], db_wavefunction[3]]

        ffft(db_engine, db_wavefunction, n_qubits)
        Ph(3 * numpy.pi / 4) | db_wavefunction

        # Flush the engine and compute expectation values and eigenvalues.
        pw_engine.flush()
        db_engine.flush()

        plane_wave_expectation_value = (
            pw_engine.backend.get_expectation_value(
                jw_dual_basis, pw_wavefunction))
        dual_basis_expectation_value = (
            db_engine.backend.get_expectation_value(
                jw_plane_wave, db_wavefunction))

        All(Measure) | pw_wavefunction
        All(Measure) | db_wavefunction

        self.assertAlmostEqual(plane_wave_expectation_value,
                               dual_basis_expectation_value)

    def test_8mode_ffft_with_external_swaps_equal_expectation_values(self):
        n_qubits = 8

        grid = Grid(dimensions=1, length=n_qubits, scale=1.0)
        dual_basis = jellium_model(grid, spinless=True, plane_wave=False)
        ffft_result = normal_ordered(dual_basis)

        # Swap 01234567 to 45670123 for fourier_transform. Additionally,
        # the FFFT's ordering is 04261537, so swap 04261537 to 01234567,
        # and then 01234567 to 45670123.
        swap_mode_list = ([1, 3, 5, 2, 4, 1, 3, 5] +
                          [3, 2, 4, 1, 3, 5, 0, 2, 4, 6, 1, 3, 5, 2, 4, 3])
        for mode in swap_mode_list:
            ffft_result = swap_adjacent_fermionic_modes(ffft_result, mode)
            ffft_result = normal_ordered(ffft_result)

        ffft_result = fourier_transform(ffft_result,
                                        grid, spinless=True)
        ffft_result = normal_ordered(ffft_result)

        # After the FFFT, swap 45670123 -> 01234567.
        swap_mode_list = [3, 2, 4, 1, 3, 5, 0, 2, 4, 6, 1, 3, 5, 2, 4, 3]
        for mode in swap_mode_list:
            ffft_result = swap_adjacent_fermionic_modes(ffft_result, mode)
            ffft_result = normal_ordered(ffft_result)

        jw_dual_basis = jordan_wigner(dual_basis)
        jw_plane_wave = jordan_wigner(ffft_result)

        # Do plane wave and dual basis calculations simultaneously.
        pw_engine = MainEngine()
        pw_wavefunction = pw_engine.allocate_qureg(n_qubits)
        db_engine = MainEngine()
        db_wavefunction = db_engine.allocate_qureg(n_qubits)

        pw_engine.flush()
        db_engine.flush()

        # Choose random state.
        state = numpy.zeros(2 ** n_qubits, dtype=complex)
        for i in range(len(state)):
            state[i] = (random.random() *
                        numpy.exp(1j * 2 * numpy.pi * random.random()))
        state /= numpy.linalg.norm(state)

        # Put randomly chosen state in the registers.
        pw_engine.backend.set_wavefunction(state, pw_wavefunction)
        db_engine.backend.set_wavefunction(state, db_wavefunction)

        prepare_logical_state(pw_wavefunction, i)
        prepare_logical_state(db_wavefunction, i)

        All(H) | [pw_wavefunction[1], pw_wavefunction[3]]
        All(H) | [db_wavefunction[1], db_wavefunction[3]]

        ffft(db_engine, db_wavefunction, n_qubits)
        Ph(3 * numpy.pi / 4) | db_wavefunction

        # Flush the engine and compute expectation values and eigenvalues.
        pw_engine.flush()
        db_engine.flush()

        plane_wave_expectation_value = (
            pw_engine.backend.get_expectation_value(
                jw_dual_basis, pw_wavefunction))
        dual_basis_expectation_value = (
            db_engine.backend.get_expectation_value(
                jw_plane_wave, db_wavefunction))

        All(Measure) | pw_wavefunction
        All(Measure) | db_wavefunction

        self.assertAlmostEqual(plane_wave_expectation_value,
                               dual_basis_expectation_value)

    def test_ffft_2d_4x4_logical_state(self):
        system_size = 4
        grid = Grid(dimensions=2, length=system_size, scale=1.0)

        fermion_operator = FermionOperator('12^ 11^ 4^ 3^')
        ffft_result = operator_2d_fft_with_reordering(
            fermion_operator, system_size)

        eng = MainEngine()
        reg = eng.allocate_qureg(system_size ** 2)
        X | reg[3]
        X | reg[4]
        X | reg[11]
        X | reg[12]
        ffft_2d(eng, reg, system_size)
        eng.flush()

        engine_wvfn = ordered_wavefunction(eng)
        operator_wvfn = numpy.zeros(2 ** (system_size ** 2), dtype=complex)
        for term in ffft_result.terms:
            i = sum(2 ** site[0] for site in term)
            operator_wvfn[i] = ffft_result.terms[term]

        All(Measure) | reg

        self.assertTrue(numpy.allclose(engine_wvfn, operator_wvfn))

    def test_ffft_2d_4x4_equal_expectation_values(self):
        system_size = 4
        n_qubits = 16
        grid = Grid(dimensions=2, length=system_size, scale=1.0)
        dual_basis = jellium_model(grid, spinless=True, plane_wave=False)
        ffft_result = operator_2d_fft_with_reordering(dual_basis, system_size)

        jw_dual_basis = jordan_wigner(dual_basis)
        jw_plane_wave = jordan_wigner(ffft_result)

        # Do plane wave and dual basis calculations together.
        pw_engine = MainEngine()
        pw_wavefunction = pw_engine.allocate_qureg(system_size ** 2)
        db_engine = MainEngine()
        db_wavefunction = db_engine.allocate_qureg(system_size ** 2)

        pw_engine.flush()
        db_engine.flush()

        # Choose random state.
        state = numpy.zeros(2 ** n_qubits, dtype=complex)
        for i in range(len(state)):
            state[i] = (random.random() *
                        numpy.exp(1j * 2 * numpy.pi * random.random()))
        state /= numpy.linalg.norm(state)

        # Put randomly chosen state in the registers.
        pw_engine.backend.set_wavefunction(state, pw_wavefunction)
        db_engine.backend.set_wavefunction(state, db_wavefunction)

        # Apply the FFFT to the dual basis wave function.
        ffft_2d(db_engine, db_wavefunction, system_size)

        # Flush the engine and compute expectation values and eigenvalues.
        pw_engine.flush()
        db_engine.flush()

        plane_wave_expectation_value = (
            pw_engine.backend.get_expectation_value(
                jw_dual_basis, pw_wavefunction))
        dual_basis_expectation_value = (
            db_engine.backend.get_expectation_value(
                jw_plane_wave, db_wavefunction))

        All(Measure) | pw_wavefunction
        All(Measure) | db_wavefunction

        self.assertAlmostEqual(plane_wave_expectation_value,
                               dual_basis_expectation_value)
