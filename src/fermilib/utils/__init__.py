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

from ._chemical_series import (make_atomic_ring,
                               make_atomic_lattice,
                               make_atom)
from ._graph import Graph, Node

from ._grid import Grid

from ._hubbard import fermi_hubbard

from ._jellium import (dual_basis_kinetic_operator,
                       dual_basis_potential_operator,
                       jellium_model,
                       jordan_wigner_dual_basis_jellium,
                       plane_wave_kinetic_operator,
                       plane_wave_potential_operator)

from ._molecular_data import MolecularData, periodic_table

from ._operator_utils import (eigenspectrum, commutator,
                              count_qubits, is_identity)

from ._plane_wave_hamiltonian import (dual_basis_external_potential_operator,
                                      inverse_fourier_transform,
                                      fourier_transform,
                                      plane_wave_hamiltonian,
                                      plane_wave_external_potential_operator,
                                      jordan_wigner_dual_basis_hamiltonian)

from ._sparse_tools import (expectation,
                            get_density_matrix,
                            get_gap,
                            get_ground_state,
                            is_hermitian,
                            jordan_wigner_sparse,
                            jw_hartree_fock_state,
                            qubit_operator_sparse,
                            sparse_eigenspectrum)

from ._trotter_error import error_bound, error_operator

from ._unitary_cc import (uccsd_operator,
                          uccsd_singlet_evolution,
                          uccsd_trotter_engine,
                          uccsd_singlet_operator,
                          uccsd_singlet_paramsize)
