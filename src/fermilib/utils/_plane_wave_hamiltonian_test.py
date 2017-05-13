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

import numpy
import unittest

from fermilib.ops import *
from fermilib.transforms import jordan_wigner
from fermilib.utils import eigenspectrum
from fermilib.utils._plane_wave_hamiltonian import *


class PlaneWaveHamiltonianTest(unittest.TestCase):

    def test_fourier_transform(self):
        n_dimensions = 1
        length_scale = 1.5
        grid_length = 3
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.5,))]
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian(
                n_dimensions, grid_length, length_scale, geometry, spinless,
                True)
            h_dual_basis = plane_wave_hamiltonian(
                n_dimensions, grid_length, length_scale, geometry, spinless,
                False)
            h_plane_wave_t = fourier_transform(
                h_plane_wave, n_dimensions, grid_length, length_scale,
                spinless)
            self.assertTrue(normal_ordered(h_plane_wave_t).isclose(
                normal_ordered(h_dual_basis)))

    def test_inverse_fourier_transform_1d(self):
        n_dimensions = 1
        length_scale = 1.5
        grid_length = 3
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.5,))]
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian(
                n_dimensions, grid_length, length_scale, geometry, spinless,
                True)
            h_dual_basis = plane_wave_hamiltonian(
                n_dimensions, grid_length, length_scale, geometry, spinless,
                False)
            h_dual_basis_t = inverse_fourier_transform(
                h_dual_basis, n_dimensions, grid_length, length_scale,
                spinless)
            self.assertTrue(normal_ordered(h_dual_basis_t).isclose(
                normal_ordered(h_plane_wave)))

    def test_inverse_fourier_transform_2d(self):
        n_dimensions = 2
        length_scale = 1.5
        grid_length = 3
        spinless = True
        geometry = [('H', (0, 0)), ('H', (0.5, 0.8))]
        h_plane_wave = plane_wave_hamiltonian(
            n_dimensions, grid_length, length_scale, geometry, spinless,
            True)
        h_dual_basis = plane_wave_hamiltonian(
            n_dimensions, grid_length, length_scale, geometry, spinless,
            False)
        h_dual_basis_t = inverse_fourier_transform(
            h_dual_basis, n_dimensions, grid_length, length_scale,
            spinless)
        self.assertTrue(normal_ordered(h_dual_basis_t).isclose(
            normal_ordered(h_plane_wave)))

    def test_u_operator_integration(self):
        n_dimensions = 1
        length_scale = 1
        grid_length = 3
        spinless_set = [True, False]
        geometry = [('H', (0,)), ('H', (0.8,))]
        for spinless in spinless_set:
            u_plane_wave = plane_wave_u_operator(
                n_dimensions, grid_length, length_scale, geometry, spinless)
            u_dual_basis = dual_basis_u_operator(
                n_dimensions, grid_length, length_scale, geometry, spinless)
            jw_u_plane_wave = jordan_wigner(u_plane_wave)
            jw_u_dual_basis = jordan_wigner(u_dual_basis)
            u_plane_wave_spectrum = eigenspectrum(jw_u_plane_wave)
            u_dual_basis_spectrum = eigenspectrum(jw_u_dual_basis)

            diff = numpy.amax(numpy.absolute(
                u_plane_wave_spectrum - u_dual_basis_spectrum))
            self.assertAlmostEqual(diff, 0)


# Run test.
if __name__ == '__main__':
    unittest.main()
