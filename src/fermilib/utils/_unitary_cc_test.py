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

"""Module to test unitary coupled cluster operators."""

from __future__ import absolute_import

import unittest
from numpy.random import randn

import fermilib
import fermilib.ops
from fermilib.ops import FermionOperator
import fermilib.utils
from fermilib.utils._unitary_cc import *

from projectq import MainEngine
from projectq.ops import (All, Measure, TimeEvolution,
                          QubitOperator,
                          X)


class UnitaryCC(unittest.TestCase):

    def test_uccsd_anti_hermitian(self):
        """Test operators are anti-Hermitian independent of inputs"""
        test_orbitals = 4

        single_amplitudes = randn(*(test_orbitals,) * 2)
        double_amplitudes = randn(*(test_orbitals,) * 4)

        generator = uccsd_operator(single_amplitudes, double_amplitudes)
        conj_generator = fermilib.ops.hermitian_conjugated(generator)

        self.assertTrue(generator.isclose(-1. * conj_generator))

    def test_uccsd_singlet_anti_hermitian(self):
        """Test that the singlet version is anti-Hermitian"""
        test_orbitals = 8
        test_electrons = 4

        packed_amplitude_size = uccsd_singlet_paramsize(test_orbitals,
                                                        test_electrons)

        packed_amplitudes = randn(int(packed_amplitude_size))

        generator = uccsd_singlet_operator(packed_amplitudes,
                                           test_orbitals,
                                           test_electrons)

        conj_generator = fermilib.ops.hermitian_conjugated(generator)

        self.assertTrue(generator.isclose(-1. * conj_generator))

    def test_uccsd_singlet_build(self):
        """Test a specific build of the UCCSD singlet operator"""
        initial_amplitudes = [-1.14941450e-08, 5.65340614e-02]
        n_orbitals = 4
        n_electrons = 2

        generator = uccsd_singlet_operator(initial_amplitudes,
                                           n_orbitals,
                                           n_electrons)

        test_generator = (0.0565340614 * FermionOperator("2^ 0 3^ 1") +
                          1.1494145e-08 * FermionOperator("1^ 3") +
                          0.0565340614 * FermionOperator("3^ 1 2^ 0") +
                          0.0565340614 * FermionOperator("2^ 0 2^ 0") +
                          1.1494145e-08 * FermionOperator("0^ 2") +
                          (-0.0565340614) * FermionOperator("1^ 3 0^ 2") +
                          (-1.1494145e-08) * FermionOperator("3^ 1") +
                          (-0.0565340614) * FermionOperator("1^ 3 1^ 3") +
                          (-0.0565340614) * FermionOperator("0^ 2 0^ 2") +
                          (-1.1494145e-08) * FermionOperator("2^ 0") +
                          0.0565340614 * FermionOperator("3^ 1 3^ 1") +
                          (-0.0565340614) * FermionOperator("0^ 2 1^ 3"))
        self.assertTrue(test_generator.isclose(generator))

    # Skip this test for now.
    @unittest.skip
    def test_projectq_filters(self):
        """Verify ProjectQ filters work as intended"""
        eng = MainEngine()
        op = QubitOperator("X0 X1 X2", -0.5)
        wavefunction = eng.allocate_qureg(4)

        command = TimeEvolution(time=1., hamiltonian=op) | wavefunction
        self.assertFalse(_identify_non_commuting(command))

        op = QubitOperator("X0 X1 X2", -0.5) + QubitOperator("Z1", 1.0)
        command = TimeEvolution(time=1., hamiltonian=op) | wavefunction

        self.assertTrue(_identify_non_commuting(command))

        op = QubitOperator("X0 X1 X2", -0.5)
        command = TimeEvolution(time=1., hamiltonian=op) | wavefunction

        self.assertFalse(_two_gate_filter(None, command))

        command = X | wavefunction

        self.assertTrue(_two_gate_filter(None, command))

    def test_simulation_energy(self):
        """Test UCCSD Singlet Energy for H2"""

        # Define H2 Hamiltonian inline
        hamiltonian = ((-0.0453222020986) * QubitOperator("X0 X1 Y2 Y3") +
                       (0.165867023964) * QubitOperator("Z0 Z3") +
                       (0.174348441706) * QubitOperator("Z2 Z3") +
                       (0.120544821866) * QubitOperator("Z0 Z2") +
                       (3.46944695195e-18) * QubitOperator("X0 Y1 X2 Y3") +
                       (0.165867023964) * QubitOperator("Z1 Z2") +
                       (0.171197748533) * QubitOperator("Z0") +
                       (-0.222785928901) * QubitOperator("Z3") +
                       (3.46944695195e-18) * QubitOperator("X0 X1 X2 X3") +
                       (0.168622191433) * QubitOperator("Z0 Z1") +
                       (0.120544821866) * QubitOperator("Z1 Z3") +
                       (3.46944695195e-18) * QubitOperator("Y0 Y1 Y2 Y3") +
                       (-0.0988639735178) * QubitOperator("") +
                       (0.171197748533) * QubitOperator("Z1") +
                       (0.0453222020986) * QubitOperator("Y0 X1 X2 Y3") +
                       (3.46944695195e-18) * QubitOperator("Y0 X1 Y2 X3") +
                       (-0.0453222020986) * QubitOperator("Y0 Y1 X2 X3") +
                       (-0.222785928901) * QubitOperator("Z2") +
                       (0.0453222020986) * QubitOperator("X0 Y1 Y2 X3"))
        hamiltonian.compress()
        compiler_engine = uccsd_trotter_engine()
        wavefunction = compiler_engine.allocate_qureg(4)
        test_amplitudes = [-1.14941450e-08, 5.65340614e-02]
        for i in range(2):
            X | wavefunction[i]
        evolution_operator = uccsd_singlet_evolution(test_amplitudes, 4, 2)
        evolution_operator | wavefunction
        compiler_engine.flush()
        energy = compiler_engine.backend.get_expectation_value(hamiltonian,
                                                               wavefunction)
        All(Measure) | wavefunction
        self.assertAlmostEqual(energy, -1.13727017463)
