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

"""Module to create and manipulate unitary coupled cluster operators."""

import itertools
import numpy

from fermilib.ops import FermionOperator
from fermilib.transforms import jordan_wigner

import projectq
import projectq.backends
import projectq.cengines
import projectq.meta
import projectq.ops
import projectq.setups
import projectq.setups.decompositions
import projectq.types


def uccsd_operator(single_amplitudes, double_amplitudes, anti_hermitian=True):
    """Create a fermionic operator that is the generator of uccsd.

    This a the most straight-forward method to generate UCCSD operators,
    however it is slightly inefficient. In particular, it parameterizes
    all possible excitations, so it represents a generalized unitary coupled
    cluster ansatz, but also does not explicitly enforce the uniqueness
    in parametrization, so it is redundant. For example there will be a linear
    dependency in the ansatz of single_amplitudes[i,j] and
    single_amplitudes[j,i].

    Args:
        single_amplitudes(list or ndarray): list of lists with each sublist
            storing a list of indices followed by single excitation amplitudes
            i.e. [[[i,j],t_ij], ...] OR [NxN] array storing single excitation
            amplitudes corresponding to
            t[i,j] * (a_i^\dagger a_j + H.C.)
        double_amplitudes(list or ndarray): list of lists with each sublist
            storing a list of indices followed by double excitation amplitudes
            i.e. [[[i,j,k,l],t_ijkl], ...] OR [NxNxNxN] array storing double
            excitation amplitudes corresponding to
            t[i,j,k,l] * (a_i^\dagger a_j a_k^\dagger a_l + H.C.)
        anti_hermitian(Bool): Flag to generate only normal CCSD operator
            rather than unitary variant, primarily for testing

    Returns:
        uccsd_generator(FermionOperator): Anti-hermitian fermion operator that
        is the generator for the uccsd wavefunction.
    """

    uccsd_generator = FermionOperator()

    # Re-format inputs (ndarrays to lists) if necessary
    if (isinstance(single_amplitudes, numpy.ndarray) or
            isinstance(double_amplitudes, numpy.ndarray)):
        single_amplitudes, double_amplitudes = convert_amplitude_format(
            single_amplitudes,
            double_amplitudes)

    # Add single excitations
    for (i, j), t_ij in single_amplitudes:
        i, j = int(i), int(j)
        uccsd_generator += FermionOperator(((i, 1), (j, 0)), t_ij)
        if anti_hermitian:
            uccsd_generator += FermionOperator(((j, 1), (i, 0)), -t_ij)

    # Add double excitations
    for (i, j, k, l), t_ijkl in double_amplitudes:
        i, j, k, l = int(i), int(j), int(k), int(l)
        uccsd_generator += FermionOperator(
            ((i, 1), (j, 0), (k, 1), (l, 0)), t_ijkl)
        if anti_hermitian:
            uccsd_generator += FermionOperator(
                ((l, 1), (k, 0), (j, 1), (i, 0)), -t_ijkl)
    return uccsd_generator


def convert_amplitude_format(single_amplitudes, double_amplitudes):
    """Re-format single_amplitudes and double_amplitudes from ndarrays to lists.

        Args:
        single_amplitudes(ndarray): [NxN] array storing single excitation
            amplitudes corresponding to t[i,j] * (a_i^\dagger a_j + H.C.)
        double_amplitudes(ndarray): [NxNxNxN] array storing double excitation
            amplitudes corresponding to
            t[i,j,k,l] * (a_i^\dagger a_j a_k^\dagger a_l + H.C.)

        Returns:
        single_amplitudes_list(list): list of lists with each sublist storing
            a list of indices followed by single excitation amplitudes
            i.e. [[[i,j],t_ij], ...]
        double_amplitudes_list(list): list of lists with each sublist storing
            a list of indices followed by double excitation amplitudes
            i.e. [[[i,j,k,l],t_ijkl], ...]
    """
    single_amplitudes_list, double_amplitudes_list = [], []

    for i, j in zip(*single_amplitudes.nonzero()):
        single_amplitudes_list.append([[i, j], single_amplitudes[i, j]])

    for i, j, k, l in zip(*double_amplitudes.nonzero()):
        double_amplitudes_list.append([[i, j, k, l],
                                      double_amplitudes[i, j, k, l]])
    return single_amplitudes_list, double_amplitudes_list


def uccsd_singlet_paramsize(n_qubits, n_electrons):
    """Determine number of independent amplitudes for singlet UCCSD

    Args:
        n_qubits(int): Number of qubits/spin-orbitals in the system
        n_electrons(int): Number of electrons in the reference state

    Returns:
        Number of independent parameters for singlet UCCSD with a single
        reference.
    """
    n_occupied = int(numpy.ceil(n_electrons / 2.))
    n_virtual = n_qubits / 2 - n_occupied

    n_single_amplitudes = n_occupied * n_virtual
    n_double_amplitudes = n_single_amplitudes ** 2
    return (n_single_amplitudes + n_double_amplitudes)


def uccsd_singlet_operator(packed_amplitudes,
                           n_qubits,
                           n_electrons):
    """Create a singlet UCCSD generator for a system with n_electrons

    This function generates a FermionOperator for a UCCSD generator designed
        to act on a single reference state consisting of n_qubits spin orbitals
        and n_electrons electrons, that is a spin singlet operator, meaning it
        conserves spin.

    Args:
        packed_amplitudes(ndarray): Compact array storing the unique single
            and double excitation amplitudes for a singlet UCCSD operator.
            The ordering lists unique single excitations before double
            excitations.
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system.

    Returns:
        uccsd_generator(FermionOperator): Generator of the UCCSD operator that
            builds the UCCSD wavefunction.
    """
    n_occupied = int(numpy.ceil(n_electrons / 2.))
    n_virtual = int(n_qubits / 2 - n_occupied)  # Virtual Spatial Orbitals
    n_t1 = int(n_occupied * n_virtual)

    t1 = packed_amplitudes[:n_t1]
    t2 = packed_amplitudes[n_t1:]

    def t1_ind(i, j):
        return i * n_occupied + j

    def t2_ind(i, j, k, l):
        return (i * n_occupied * n_virtual * n_occupied +
                j * n_virtual * n_occupied +
                k * n_occupied +
                l)

    uccsd_generator = FermionOperator()

    spaces = range(n_virtual), range(n_occupied), range(2)

    for i, j, s in itertools.product(*spaces):
        uccsd_generator += FermionOperator(
            (
                (2 * (i + n_occupied) + s, 1),
                (2 * j + s, 0),
            ),
            coefficient=t1[t1_ind(i, j)])

        uccsd_generator += FermionOperator(
            (
                (2 * j + s, 1),
                (2 * (i + n_occupied) + s, 0),
            ),
            coefficient=-t1[t1_ind(i, j)])

    for i, j, s, i2, j2, s2 in itertools.product(*spaces, repeat=2):
        uccsd_generator += FermionOperator((
            (2 * (i + n_occupied) + s, 1),
            (2 * j + s, 0),
            (2 * (i2 + n_occupied) + s2, 1),
            (2 * j2 + s2, 0)),
            t2[t2_ind(i, j, i2, j2)])

        uccsd_generator += FermionOperator((
            (2 * j2 + s2, 1),
            (2 * (i2 + n_occupied) + s2, 0),
            (2 * j + s, 1),
            (2 * (i + n_occupied) + s, 0)),
            -t2[t2_ind(i, j, i2, j2)])

    return uccsd_generator


def uccsd_evolution(fermion_generator, fermion_transform=jordan_wigner):
    """Create a ProjectQ evolution operator for a UCCSD circuit

    Args:
        fermion_generator(FermionOperator): UCCSD generator to evolve.
        fermion_transform(fermilib.transform): The transformation that
            defines the mapping from Fermions to QubitOperator.

    Returns:
        evoution_operator(projectq.ops.TimeEvolution): The unitary operator
            that constructs the UCCSD state.
    """

    # Transform generator to qubits
    qubit_generator = fermion_transform(fermion_generator)

    # Cast to real part only for compatibility with current ProjectQ routine
    for key in qubit_generator.terms:
        qubit_generator.terms[key] = float(qubit_generator.terms[key].imag)
    qubit_generator.compress()

    # Allocate wavefunction and act evolution on gate according to compilation
    evolution_operator = (
        projectq.ops.TimeEvolution(time=1., hamiltonian=qubit_generator))

    return evolution_operator


def uccsd_singlet_evolution(packed_amplitudes, n_qubits, n_electrons,
                            fermion_transform=jordan_wigner):
    """Create a ProjectQ evolution operator for a UCCSD singlet circuit

    Args:
        packed_amplitudes(ndarray): Compact array storing the unique single
            and double excitation amplitudes for a singlet UCCSD operator.
            The ordering lists unique single excitations before double
            excitations.
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system
        fermion_transform(fermilib.transform): The transformation that
            defines the mapping from Fermions to QubitOperator.

    Returns:
        evoution_operator(projectq.ops.TimeEvolution): The unitary operator
            that constructs the UCCSD singlet state.
    """
    # Build UCCSD generator
    fermion_generator = uccsd_singlet_operator(packed_amplitudes,
                                               n_qubits,
                                               n_electrons)

    evolution_operator = uccsd_evolution(fermion_generator,
                                         fermion_transform)

    return evolution_operator


def _identify_non_commuting(cmd):
    """Recognize all TimeEvolution gates with >1 terms that don't all commute.

    This is a filter function for use with ProjectQ that flags terms as
        non-commuting so they may be handled by a different factorization
        routine.

    Args:
        cmd(projectq.command): A command from ProjectQ

    Returns:
        (bool) Depending on whether the terms are determined to commute or not
    """
    hamiltonian = cmd.gate.hamiltonian
    if len(hamiltonian.terms) == 1:
        return False
    else:
        id_op = projectq.ops.QubitOperator((), 0.0)
        for term in hamiltonian.terms:
            test_op = projectq.ops.QubitOperator(term, hamiltonian.terms[term])
            for other in hamiltonian.terms:
                other_op = (
                    projectq.ops.QubitOperator(other,
                                               hamiltonian.terms[other]))
                commutator = test_op * other_op - other_op * test_op
                if not commutator.isclose(id_op,
                                          rel_tol=1e-9,
                                          abs_tol=1e-9):
                    return True
    return False


def _non_adjacent_filter(self, cmd, qubit_graph, flip=False):
    """A ProjectQ filter to identify when swaps are needed on a graph

    This flags any gates that act on two non-adjacent qubits with respect to
    the qubit_graph that has been given

    Args:
        self(Dummy): Dummy parameter to meet function specification.
        cmd(projectq.command): Command to be checked for decomposition into
            additional swap gates.
        qubit_graph(Graph): Graph object specifying connectivity of
            qubits. The values of the nodes of this graph are unique qubit ids.
        flip(Bool): Flip for switching if identifying a gate is in this class
            by true or false.  Designed to meet the specification of ProjectQ
            InstructionFilter and DecompositionRule with one function.

    Returns:
        bool: When flip is False, this returns True when a 2 qubit command
            acts on non-adjacent qubits or when it acts only on a single qubit.
            This is reversed when flip is used.

    """
    if qubit_graph is None:
        return True ^ flip

    total_qubits = (cmd.control_qubits +
                    [item for qureg in cmd.qubits for item in qureg])

    # Check for non-connected gate on 2 qubits
    if ((len(total_qubits) == 1) or
            (len(total_qubits) == 2 and
             qubit_graph.is_adjacent(
                 qubit_graph.find_index(total_qubits[0].id),
                 qubit_graph.find_index(total_qubits[1].id)))):
        return True ^ flip
    return False ^ flip


def _direct_graph_swap(cmd, qubit_graph):
    """Define a naive direct swap sequence to respect qubit_graph connectivity

    Uses the connectivity of qubit_graph to find the shortest path between
    two non-adjacent qubits, and swaps/unswaps qubits appropriately.  Baseline
    for more sophisticated algorithms

    Args:
        cmd(projectq.command): A command from ProjectQ that needs to be
            broken down due to non-adjacent terms
        qubit_graph(Graph): Graph object specifying connectivity of qubits.
            The values of the nodes of this graph are unique qubit ids
    """
    total_qubits = (cmd.control_qubits +
                    [item for qureg in cmd.qubits for item in qureg])

    gate = cmd.gate
    engine = cmd.engine
    graph_path = qubit_graph.shortest_path(
        qubit_graph.find_index(total_qubits[0].id),
        qubit_graph.find_index(total_qubits[1].id))
    swap_path = [(graph_path[i], graph_path[i + 1])
                 for i in range(len(graph_path) - 2)]

    # SWAP qubit 1 into position adjacent to qubit 2
    for pair in swap_path:
        projectq.ops.Swap | (projectq.types.
                             WeakQubitRef(engine,
                                          qubit_graph.nodes[pair[0]].value),
                             projectq.types.
                             WeakQubitRef(engine,
                                          qubit_graph.nodes[pair[1]].value))

    # Perform original gate
    if len(cmd.control_qubits) > 0:
        projectq.ops.C(gate) | (projectq.types.
                                WeakQubitRef(engine,
                                             qubit_graph.
                                             nodes[graph_path[-2]].value),
                                total_qubits[1])
    else:
        gate | (projectq.types.
                WeakQubitRef(engine,
                             qubit_graph.nodes[graph_path[-2]].value),
                total_qubits[1])

    # Reverse the swaps to put qubits back in place
    for pair in reversed(swap_path):
        projectq.ops.Swap | (projectq.types.
                             WeakQubitRef(engine,
                                          qubit_graph.nodes[pair[0]].value),
                             projectq.types.
                             WeakQubitRef(engine,
                                          qubit_graph.nodes[pair[1]].value))


def _first_order_trotter(cmd):
    """Define a Trotter splitting for non-commuting Pauli in ProjectQ

    This routine defines a first-order Trotter splitting to be applied to
    time evolution operators in ProjectQ.

    Args:
        cmd(projectq.command): A command from ProjectQ that needs to be
            factorized due to non-commuting time evolution terms
    """
    qureg = cmd.qubits
    eng = cmd.engine
    hamiltonian = cmd.gate.hamiltonian
    time = cmd.gate.time
    with projectq.meta.Control(eng, cmd.control_qubits):
        # First order Trotter splitting
            for term in hamiltonian.terms:
                ind_operator = (projectq.
                                ops.
                                QubitOperator(term, hamiltonian.terms[term]))
                projectq.ops.TimeEvolution(time, ind_operator) | qureg


def _two_gate_filter(self, cmd):
    """A ProjectQ filter to flag TimeEvolution operators for decomposition

    This flags any gates which act on more than 2 qubits or are time evolution
    operators to be decomposed into a base library of gates for simulation
    within ProjectQ.

    Args:
        self(Dummy): Dummy parameter to meet function specification.
        cmd(projectq.command): Command to be checked for decomposition into
            one- and two- qubit gates.

    """
    if ((not isinstance(cmd.gate, projectq.ops.TimeEvolution)) and
        (len(cmd.qubits[0]) <= 2 or
            isinstance(cmd.gate, projectq.ops.ClassicalInstructionGate))):
        return True
    return False


def uccsd_trotter_engine(compiler_backend=projectq.backends.Simulator(),
                         qubit_graph=None):
    """Define a ProjectQ compiler engine that is common for use with UCCSD

    This defines a ProjectQ compiler engine that decomposes time evolution
    gates using a first order Trotter decomposition on non-commuting gates
    down to a base gate decomposition.

    Args:
        compiler_backend(projectq.backend): Define the backend on the
            circuit compiler, so that it may either simulate gates numerically
            or alternatively print a gate sequence, e.g. using
            projectq.backends.CommandPrinter()
        qubit_graph(Graph): Graph object specifying connectivity of qubits.
            The values of the nodes of this unique qubit ids.  If None,
            all-to-all connectivity is assumed.

    Returns:
        projectq.cengine that is the compiler engine set up with these
            rules and decompostions.
    """
    rule_set = (
        projectq.cengines.
        DecompositionRuleSet(modules=[projectq.setups.decompositions]))

    # Set rules for splitting non-commuting operators
    trotter_rule_set = (projectq.cengines.DecompositionRule(
        gate_class=projectq.ops.TimeEvolution,
        gate_decomposer=_first_order_trotter,
        gate_recognizer=_identify_non_commuting))
    rule_set.add_decomposition_rule(trotter_rule_set)

    # Set rules for 2 qubit gates that act on non-adjacent qubits
    if qubit_graph is not None:
        connectivity_rule_set = (
            projectq.cengines.DecompositionRule(
                gate_class=projectq.ops.NOT.__class__,
                gate_decomposer=(lambda x: _direct_graph_swap(x, qubit_graph)),
                gate_recognizer=(lambda x: _non_adjacent_filter(None, x,
                                                                qubit_graph,
                                                                True))))
        rule_set.add_decomposition_rule(connectivity_rule_set)

    # Build the full set of engines that will be applied to qubits
    replacer = projectq.cengines.AutoReplacer(rule_set)
    compiler_engine_list = [replacer,
                            projectq.
                            cengines.
                            InstructionFilter(
                                lambda x, y:
                                (_non_adjacent_filter(x, y, qubit_graph) and
                                 _two_gate_filter(x, y))),
                            projectq.cengines.LocalOptimizer(5)]

    # Start the compiler engine with these rules
    compiler_engine = (
        projectq.MainEngine(backend=compiler_backend,
                            engine_list=compiler_engine_list))
    return compiler_engine
