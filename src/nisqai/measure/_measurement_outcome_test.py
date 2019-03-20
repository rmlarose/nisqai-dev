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

from pyquil import Program, get_qc
from pyquil.gates import X, H, MEASURE

from nisqai.measure._measurement_outcome import MeasurementOutcome


def get_all_zeros_outcome(nqubits, nshots):
    """Helper function that returns the outcome of all zeros.

    Args:
        nqubits : int
            Number of qubits in the circuit.

        nshots : int
            Number of shots to simulate the circuit.
    """
    prog = Program()
    creg = prog.declare("ro", memory_type="BIT", memory_size=nqubits)

    prog += [MEASURE(q, creg[q]) for q in range(nqubits)]

    prog.wrap_in_numshots_loop(nshots)

    computer = get_qc("{}q-qvm".format(nqubits))

    return computer.run(prog)


def get_all_ones_outcome(nqubits, nshots):
    """Helper function that returns the outcome of all ones.

    Args:
        nqubits : int
            Number of qubits in the circuit.

        nshots : int
            Number of shots to simulate the circuit.
    """
    prog = Program()
    creg = prog.declare("ro", memory_type="BIT", memory_size=nqubits)

    prog += [X(q) for q in range(nqubits)]
    prog += [MEASURE(q, creg[q]) for q in range(nqubits)]

    prog.wrap_in_numshots_loop(nshots)

    computer = get_qc("{}q-qvm".format(nqubits))

    return computer.run(prog)


def test_basic():
    """Tests that a MeasurementOutcome can be instantiated."""
    # get an outcome from simulating a circuit
    result = get_all_ones_outcome(4, 10)

    # create a MeasurementOutcome
    outcome = MeasurementOutcome(result)

    # trivial check
    assert (outcome.raw_outcome == result).all()


def test_num_qubits():
    """Tests that a MeasurementOutcome has the right qubit number."""
    # number of qubits
    nqubits = 4

    # get an outcome from simulating a circuit
    result = get_all_ones_outcome(nqubits, 10)

    # create a MeasurementOutcome
    outcome = MeasurementOutcome(result)

    # trivial check
    assert outcome.num_qubits == nqubits


def test_num_shots():
    """Tests that a MeasurementOutcome has the right number of shots."""
    # number of qubits
    nqubits = 4

    # number of shots
    nshots = 40

    # get an outcome from simulating a circuit
    result = get_all_ones_outcome(nqubits, nshots)

    # create a MeasurementOutcome
    outcome = MeasurementOutcome(result)

    # trivial check
    assert outcome.shots == nshots


def test_get_item():
    """Tests getting an item from a measurement outcome."""
    # number of qubits
    nqubits = 5

    # number of shots
    nshots = 40

    # get an outcome from simulating a circuit
    result = get_all_ones_outcome(nqubits, nshots)

    # create a MeasurementOutcome
    outcome = MeasurementOutcome(result)

    assert len(outcome[0]) == 5


def test_len():
    """Tests the length of a measurement outcome."""
    # get an outcome from simulating a circuit
    result = get_all_ones_outcome(nqubits=2, nshots=1000)

    # create a MeasurementOutcome
    outcome = MeasurementOutcome(result)

    assert len(outcome) == 1000


def test_as_int():
    """Tests the integer value of bit strings is correct."""
    # get some measurement outcomes
    zeros = MeasurementOutcome(get_all_zeros_outcome(nqubits=2, nshots=20))
    ones = MeasurementOutcome(get_all_ones_outcome(nqubits=2, nshots=20))

    # checks for zeros
    assert type(zeros.as_int(0)) == int
    assert zeros.as_int(0) == 0

    # checks for ones
    assert type(ones.as_int(0)) == int
    assert ones.as_int(0) == 3


def test_as_int_big_int():
    """Tests the integer value of bit strings for large integers."""
    # get a measurement outcome
    ones = MeasurementOutcome(get_all_ones_outcome(nqubits=10, nshots=20))

    # checks for ones
    assert type(ones.as_int(0)) == int
    assert ones.as_int(0) == 2**10 - 1


if __name__ == "__main__":
    test_basic()
    test_num_qubits()
    test_num_shots()
    test_get_item()
    test_len()
    test_as_int()
    test_as_int_big_int()
    print("All tests for MeasurementOutcome class passed.")

