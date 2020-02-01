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

"""Unit tests for BaseAnsatz class."""

from pyquil import gates

from nisqai.layer._base_ansatz import BaseAnsatz
from numpy import array, pi, exp


def test_basic():
    """Basic test for BaseAnsatz."""
    n = 3
    b = BaseAnsatz(n)
    assert b.num_qubits == n


def test_add():
    """Add two BaseAnsatz's and make sure the result is correct."""
    a = BaseAnsatz(3)
    b = BaseAnsatz(3)
    
    a.add_layer(gates.X)
    b.add_layer(gates.Y)
    
    print("a + b =\n", a + b)


def test_depth_no_gates():
    """Checks that depth is being calculated correctly from compiled Quil program
    with no gates.

    REQUIRES
        Quil compiler (quilc) to be running.
    """
    ansatz = BaseAnsatz(num_qubits=10)
    computer = "10q-qvm"
    assert ansatz.depth(computer) == 0


def test_depth_z_rotation():
    """Checks that depth is being calculated correctly from compiled Quil programs
    with Z rotations on each qubit.

    REQUIRES
        Quil compiler (quilc) to be running.
    """
    for nqubits in range(1, 5):
        ansatz = BaseAnsatz(num_qubits=nqubits)
        ansatz.add_layer(gates.Z)
        computer = f"{nqubits}q-qvm"
        print(ansatz.compile(computer))
        assert ansatz.depth(computer) == nqubits * 3  # Quil Compiler does Z = Z P P^\dagger where P = Rx(pi / 2)


def test_depth():
    """Checks that depth is being calculated correctly from compiled Quil program.

    REQUIRES
        Quil compiler (quilc) to be running.
    """
    ansatz = BaseAnsatz(2)
    ansatz.circuit += [gates.X(0), gates.RZ(pi/4, 1)]
    computer = "2q-qvm"
    assert ansatz.depth(computer) == 7


def test_depth_identity():
    """Checks that depth is being calculated correctly for hardware gates.

    REQUIRES
        Quil compiler (quilc) to be running.
    """
    ansatz = BaseAnsatz(1)
    ansatz.circuit.inst(gates.RZ(0, 0))
    assert ansatz.depth("1q-qvm") == 0


def test_compile_small():
    """Tests correct compilation for a small circuit."""
    # correct output string
    correct = ('RX(pi/2) 0'
               + '\nHALT\n')
    ansatz = BaseAnsatz(2)
    ansatz.circuit.inst(gates.RX(pi / 2, 0))
    compiled_circuit = ansatz.compile("6q-qvm")
    print(compiled_circuit.__str__())
    print(correct)
    assert compiled_circuit.__str__() == correct


def test_compile_defined_gate():
    """Tests compiling a program with a defined gate."""
    # get an ansatz
    ansatz = BaseAnsatz(1)

    # get a matrix for a gate
    gate = array([[1.0, 0], [0, exp(1j * 2 * pi / 32)]])

    # define a gate
    ansatz.circuit.defgate("G", gate)

    # add the gate
    ansatz.circuit += ("G", 0)

    # compile the program
    compiled = ansatz.compile("1q-qvm")

    print(compiled)


def test_add_at():
    """Tests the BaseAnsatz.add_at method for a small circuit."""
    ansatz = BaseAnsatz(4)
    gate = gates.X
    indices = [0, 2]
    ansatz.add_at(gate, indices)
    print(ansatz)


if __name__ == "__main__":
    test_basic()
    test_add()
    test_depth()
    test_depth_empty()
    test_compile_small()
    test_compile_defined_gate()
    test_add_at()
    print("All tests for BaseAnsatz class passed.")
