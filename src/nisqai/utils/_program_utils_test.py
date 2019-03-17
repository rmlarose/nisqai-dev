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

from nisqai.utils._program_utils import order, make_ascii_circuit
from pyquil import Program, gates
from pyquil.quil import Pragma
import numpy as np


def test_order_basic():
    """Tests an ordered program is in the nominal form."""
    p = Program(gates.H(0))
    p.declare("ro")
    print(order(p))


def test_ascii_circuit_basic():
    """Tests drawing a program on one qubit."""
    prog = Program(
        [gates.H(0),
         gates.X(0)]
    )
    drawing = make_ascii_circuit(prog, 2)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_multiple_qubits():
    """Tests drawing a program with single qubit gates on multiple qubits."""
    prog = Program(
        [gates.H(0),
         gates.X(1),
         gates.Z(0),
         gates.H(1)]
    )
    drawing = make_ascii_circuit(prog, 2)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_two_qubit_gates():
    """Tests drawing a program with two qubit gates."""
    prog = Program(
        [gates.H(0),
         gates.CNOT(0, 1)]
    )
    drawing = make_ascii_circuit(prog, 2)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_with_definted_gate():
    """Tests drawing a program with defined gate."""
    prog = Program()
    Sgate = np.array([[1, 0],
                      [0, 1]])
    prog.defgate("S", Sgate)
    prog += ("S", 0)
    prog += gates.CNOT(0, 1)
    drawing = make_ascii_circuit(prog)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_with_pramga1():
    """Tests drawing a program with a pramga."""
    prog = Program()
    prog += Program(Pragma('INITIAL_REWIRING', ['"GREEDY"']))
    prog += gates.CNOT(0, 1)
    drawing = make_ascii_circuit(prog)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_with_pramga2():
    """Tests drawing a program with a different pramga."""
    prog = Program(
        [gates.H(0),
         gates.CNOT(0, 1)]
    )
    prog += Pragma('DELAY', [1], str(2))
    drawing = make_ascii_circuit(prog)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_with_measurement():
    """Tests drawing a program with a measurement."""
    prog = Program(
        [gates.H(0),
         gates.CNOT(0, 1)]
    )
    # adds a measurement to register 'ro'
    creg = prog.declare("ro", memory_size=1)
    prog += (gates.MEASURE(0, creg[0]))
    drawing = make_ascii_circuit(prog)

    assert type(drawing) == str
    print(drawing)


if __name__ == "__main__":
    test_order_basic()
    test_ascii_circuit_basic()
    test_ascii_circuit_multiple_qubits()
    test_ascii_circuit_two_qubit_gates()
    test_ascii_circuit_with_definted_gate()
    test_ascii_circuit_with_pramga1()
    test_ascii_circuit_with_pramga2()
    test_ascii_circuit_with_measurement()
    print("All unit tests for program_utils passed.")
