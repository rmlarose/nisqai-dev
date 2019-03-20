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

from numpy import array

from pyquil import Program, gates
from pyquil.quil import Pragma

from nisqai.utils._program_utils import order, ascii_drawer
from nisqai.layer import ProductAnsatz



def test_order_basic():
    """Tests an ordered program is in the nominal form."""
    p = Program(gates.H(0))
    p.declare("ro")
    print(order(p))


def test_ascii_circuit_basic():
    """Tests drawing a program on one qubit.

    Should display:

    0 |0> --H----X--
    """
    prog = Program(
        [gates.H(0),
         gates.X(0)]
    )
    drawing = ascii_drawer(prog)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_multiple_qubits():
    """Tests drawing a program with single qubit gates on multiple qubits.

    Should display:

    0 |0> --H---------Z-------
    1 |0> -------X---------H--
    """
    prog = Program(
        [gates.H(0),
         gates.X(1),
         gates.Z(0),
         gates.H(1)]
    )
    drawing = ascii_drawer(prog)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_two_qubit_gates():
    """Tests drawing a program with two qubit gates.

    Should display:

    0 |0> --H----CNOT'--
    1 |0> -------CNOT --
    """
    prog = Program(
        [gates.H(0),
         gates.CNOT(0, 1)]
    )
    drawing = ascii_drawer(prog, 2)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_with_defined_gate():
    """Tests drawing a program with defined gate.

    Should display:

    0 |0> --S----CNOT'--
    1 |0> -------CNOT --
    """
    prog = Program()
    sgate = array([[1, 0],
                   [0, 1]])
    prog.defgate("S", sgate)
    prog += ("S", 0)
    prog += gates.CNOT(0, 1)
    drawing = ascii_drawer(prog)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_with_pragma1():
    """Tests drawing a program with a pragma.

    Should display:

    0 |0> --CNOT'--
    1 |0> --CNOT --

    """
    prog = Program()
    prog += Program(Pragma('INITIAL_REWIRING', ['"GREEDY"']))
    prog += gates.CNOT(0, 1)
    drawing = ascii_drawer(prog)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_with_pragma2():
    """Tests drawing a program with a different pramga.

    Should display:

    0 |0> --H----CNOT'--
    1 |0> -------CNOT --

    """
    prog = Program(
        [gates.H(0),
         gates.CNOT(0, 1)]
    )
    prog += Pragma('DELAY', [1], str(2))
    drawing = ascii_drawer(prog)

    assert type(drawing) == str
    print(drawing)


def test_ascii_circuit_with_measurement():
    """Tests drawing a program with a measurement.

    Should display:

    0 |0> --H----CNOT'----MSR--
    1 |0> -------CNOT ---------

    """
    prog = Program(
        [gates.H(0),
         gates.CNOT(0, 1)]
    )
    # adds a measurement to register 'ro'
    creg = prog.declare("ro", memory_size=1)
    prog += (gates.MEASURE(0, creg[0]))
    drawing = ascii_drawer(prog)

    assert type(drawing) == str
    print(drawing)


def test_cnot_non_adjacent_qubits():
    """Tests drawing a program with a CNOT gate between non-adjacent qubits.

    Should display:

    0 |0> --H----CNOT'----CNOT'----MSR--
    1 |0> -------CNOT ------------------
    2 |0> ----------------CNOT ---------

    """
    prog = Program(
        [gates.H(0),
         gates.CNOT(0, 1),
         gates.CNOT(0, 2)]
    )
    # adds a measurement to register 'ro'
    creg = prog.declare("ro", memory_size=1)
    prog += (gates.MEASURE(0, creg[0]))
    drawing = ascii_drawer(prog)

    assert type(drawing) == str
    print(drawing)


def test_empty_qubit():
    """Tests drawing a program with no operations on a qubit.

    Should display:

    0 |0> --H----CNOT'----MSR--
    2 |0> -------CNOT ---------
    """
    prog = Program(
        [gates.H(0),
         gates.CNOT(0, 2)]
    )
    # adds a measurement to register 'ro'
    creg = prog.declare("ro", memory_size=1)
    prog += (gates.MEASURE(0, creg[0]))
    drawing = ascii_drawer(prog)

    assert type(drawing) == str
    print(drawing)


def test_product_ansatz():
    """Creates a ProductAnsatz and draws it."""
    # get an ansatz
    ansatz = ProductAnsatz(4)

    # make the drawing
    drawing = ascii_drawer(ansatz.circuit)

    # show the drawing
    print(drawing)


if __name__ == "__main__":
    test_order_basic()
    test_ascii_circuit_basic()
    test_ascii_circuit_multiple_qubits()
    test_ascii_circuit_two_qubit_gates()
    test_ascii_circuit_with_defined_gate()
    test_ascii_circuit_with_pragma1()
    test_ascii_circuit_with_pragma2()
    test_ascii_circuit_with_measurement()
    test_cnot_non_adjacent_qubits()
    test_empty_qubit()
    test_product_ansatz()
    print("All unit tests for program_utils passed.")
