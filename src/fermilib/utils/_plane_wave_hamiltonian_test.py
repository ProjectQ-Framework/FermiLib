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

"""Tests for plane_wave_hamiltonian.py"""
from __future__ import absolute_import

import unittest

import numpy

from fermilib.ops import normal_ordered
from fermilib.transforms import jordan_wigner
from fermilib.utils import eigenspectrum, Grid
from fermilib.utils._plane_wave_hamiltonian import (
    plane_wave_hamiltonian,
    fourier_transform,
    inverse_fourier_transform,
    plane_wave_u_operator,
    dual_basis_u_operator,
    jordan_wigner_dual_basis_hamiltonian,
)


class PlaneWaveHamiltonianTest(unittest.TestCase):

    def test_fourier_transform(self):
        grid = Grid(dimensions=1, scale=1.5, length=3)
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.5,))]
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian(
                grid, geometry, spinless, True)
            h_dual_basis = plane_wave_hamiltonian(
                grid, geometry, spinless, False)
            h_plane_wave_t = fourier_transform(h_plane_wave, grid, spinless)
            self.assertTrue(normal_ordered(h_plane_wave_t).isclose(
                normal_ordered(h_dual_basis)))

    def test_inverse_fourier_transform_1d(self):
        grid = Grid(dimensions=1, scale=1.5, length=4)
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.5,))]
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian(
                grid, geometry, spinless, True)
            h_dual_basis = plane_wave_hamiltonian(
                grid, geometry, spinless, False)
            h_dual_basis_t = inverse_fourier_transform(
                h_dual_basis, grid, spinless)
            self.assertTrue(normal_ordered(h_dual_basis_t).isclose(
                normal_ordered(h_plane_wave)))

    def test_inverse_fourier_transform_2d(self):
        grid = Grid(dimensions=2, scale=1.5, length=3)
        spinless = True
        geometry = [('H', (0, 0)), ('H', (0.5, 0.8))]
        h_plane_wave = plane_wave_hamiltonian(grid, geometry, spinless, True)
        h_dual_basis = plane_wave_hamiltonian(grid, geometry, spinless, False)
        h_dual_basis_t = inverse_fourier_transform(
            h_dual_basis, grid, spinless)
        self.assertTrue(normal_ordered(h_dual_basis_t).isclose(
            normal_ordered(h_plane_wave)))

    def test_plane_wave_hamiltonian_integration(self):
        length_set = [3, 4]
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.8,))]
        for l in length_set:
            for spinless in spinless_set:
                grid = Grid(dimensions=1, scale=1.0, length=l)
                h_plane_wave = plane_wave_hamiltonian(grid, geometry, spinless,
                                                      True)
                h_dual_basis = plane_wave_hamiltonian(grid, geometry, spinless,
                                                      False)
                jw_h_plane_wave = jordan_wigner(h_plane_wave)
                jw_h_dual_basis = jordan_wigner(h_dual_basis)
                h_plane_wave_spectrum = eigenspectrum(jw_h_plane_wave)
                h_dual_basis_spectrum = eigenspectrum(jw_h_dual_basis)

                diff = numpy.amax(numpy.absolute(
                    h_plane_wave_spectrum - h_dual_basis_spectrum))
                self.assertAlmostEqual(diff, 0)

    def test_jordan_wigner_dual_basis_hamiltonian(self):
        grid = Grid(dimensions=2, length=3, scale=1.)
        spinless = True
        geometry = [('H', (0, 0)), ('H', (0.5, 0.8))]

        fermion_hamiltonian = plane_wave_hamiltonian(grid, geometry, spinless,
                                                     False)
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

        test_hamiltonian = jordan_wigner_dual_basis_hamiltonian(grid, geometry,
                                                                spinless)
        self.assertTrue(test_hamiltonian.isclose(qubit_hamiltonian))


# Run test.
if __name__ == '__main__':
    unittest.main()
