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

"""Tests for molecular_data."""
from __future__ import absolute_import

import numpy.random
import scipy.linalg
import unittest

from fermilib.config import *
from fermilib.utils import *
from fermilib.utils._molecular_data import *


class MolecularDataTest(unittest.TestCase):

    def setUp(self):
        self.geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        filename = os.path.join(THIS_DIRECTORY, 'data', 'H2_sto-3g_singlet')
        self.molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity, filename=filename)
        self.molecule.load()

    def test_name_molecule(self):
        charge = 0
        correct_name = 'H2_sto-3g_singlet'
        computed_name = name_molecule(self.geometry,
                                      self.basis,
                                      self.multiplicity,
                                      charge,
                                      description=None)
        self.assertEqual(correct_name, computed_name)
        self.assertEqual(correct_name, self.molecule.name)

    def test_geometry_from_file(self):
        water_geometry = [('O', (0., 0., 0.)),
                          ('H', (0.757, 0.586, 0.)),
                          ('H', (-.757, 0.586, 0.))]
        filename = os.path.join(THIS_DIRECTORY, 'data', 'geometry_example.txt')
        test_geometry = geometry_from_file(filename)
        for atom in range(3):
            self.assertAlmostEqual(water_geometry[atom][0],
                                   test_geometry[atom][0])
            for coordinate in range(3):
                self.assertAlmostEqual(water_geometry[atom][1][coordinate],
                                       test_geometry[atom][1][coordinate])

    def test_save_load(self):
        n_atoms = self.molecule.n_atoms
        self.molecule.n_atoms += 1
        self.assertEqual(self.molecule.n_atoms, n_atoms + 1)
        self.molecule.load()
        self.assertEqual(self.molecule.n_atoms, n_atoms)

    def test_dummy_save(self):

        # Make fake molecule.
        filename = os.path.join(THIS_DIRECTORY, 'data', 'dummy_molecule')
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = '6-31g*'
        multiplicity = 7
        charge = -1
        description = 'fermilib_forever'
        molecule = MolecularData(geometry, basis, multiplicity,
                                 charge, description, filename)

        # Make some attributes to save.
        molecule.n_orbitals = 10
        molecule.n_qubits = 10
        molecule.nuclear_repulsion = -12.3
        molecule.hf_energy = 88.
        molecule.canonical_orbitals = [1, 2, 3, 4]
        molecule.orbital_energies = [5, 6, 7, 8]
        molecule.orbital_overlaps = [1, 2, 3, 4]
        molecule.one_body_integrals = [5, 6, 7, 8]
        molecule.mp2_energy = -12.
        molecule.cisd_energy = 32.
        molecule.cisd_one_rdm = numpy.arange(10)
        molecule.fci_energy = 232.
        molecule.fci_one_rdm = numpy.arange(11)
        molecule.ccsd_energy = 88.

        # Save molecule.
        molecule.save()

        # Change attributes and load.
        molecule.ccsd_energy = -2.232

        # Load molecule.
        new_molecule = MolecularData(filename=filename)
        molecule.load()

        # Check CCSD energy.
        self.assertAlmostEqual(new_molecule.ccsd_energy, molecule.ccsd_energy)
        self.assertAlmostEqual(molecule.ccsd_energy, 88.)

    def test_energies(self):
        self.assertAlmostEqual(self.molecule.hf_energy, -1.1167, places=4)
        self.assertAlmostEqual(self.molecule.mp2_energy, -1.1299, places=4)
        self.assertAlmostEqual(self.molecule.cisd_energy, -1.1373, places=4)
        self.assertAlmostEqual(self.molecule.ccsd_energy, -1.1373, places=4)
        self.assertAlmostEqual(self.molecule.ccsd_energy, -1.1373, places=4)

    def test_rdm_and_rotation(self):

        # Compute total energy from RDM.
        molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()
        molecular_rdm = self.molecule.get_molecular_rdm()
        total_energy = molecular_rdm.expectation(molecular_hamiltonian)
        self.assertAlmostEqual(total_energy, self.molecule.cisd_energy)

        # Build random rotation with correction dimension.
        num_spatial_orbitals = self.molecule.n_orbitals
        rotation_generator = numpy.random.randn(
            num_spatial_orbitals, num_spatial_orbitals)
        rotation_matrix = scipy.linalg.expm(
            rotation_generator - rotation_generator.T)

        # Compute total energy from RDM under some basis set rotation.
        molecular_rdm.rotate_basis(rotation_matrix)
        molecular_hamiltonian.rotate_basis(rotation_matrix)
        total_energy = molecular_rdm.expectation(molecular_hamiltonian)
        self.assertAlmostEqual(total_energy, self.molecule.cisd_energy)

    def test_get_up_down_electrons(self):
        largest_atom = 20
        for n_electrons in range(1, largest_atom):

            # Make molecule.
            basis = 'sto-3g'
            atom_name = periodic_table[n_electrons]
            molecule = make_atom(atom_name, basis)

            # Get expected alpha and beta.
            spin = periodic_polarization[n_electrons] / 2.
            multiplicity = int(2 * spin + 1)
            expected_alpha = n_electrons / 2 + (multiplicity - 1)
            expected_beta = n_electrons / 2 - (multiplicity - 1)

            # Test.
            self.assertAlmostEqual(molecule.get_n_alpha_electrons(),
                                   expected_alpha)
            self.assertAlmostEqual(molecule.get_n_beta_electrons(),
                                   expected_beta)


if __name__ == '__main__':
    unittest.main()
