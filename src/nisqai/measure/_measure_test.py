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

from nisqai.measure._measure import Measurement


def test_measure_all():
    """Tests measurements on all qubits."""
    measure = Measurement(9, range(9))
    assert measure.num_qubits == 9
    assert measure.num_measurements == 9
    assert measure.measured_qubits == list(range(9))
    assert len(measure.circuit.instructions) == 10


def test_measure_single():
    """Tests measurements on a single qubit."""
    measure = Measurement(10, [1])
    assert measure.num_qubits == 10
    assert measure.num_measurements == 1
    assert measure.measured_qubits == [1]
    assert len(measure.circuit.instructions) == 2


def test_measure_some():
    """Tests measurements on a subset of qubits."""
    measure = Measurement(10, [3, 5, 8])
    assert measure.num_qubits == 10
    assert measure.num_measurements == 3
    assert measure.measured_qubits == [3, 5, 8]


def test_measure_tuple():
    """Tests measurements on all qubits with a tuple as input."""
    measure = Measurement(4, (0, 1, 2, 3))
    assert measure.num_qubits == 4
    assert measure.measured_qubits == [0, 1, 2, 3]
    assert measure.num_measurements == 4


def test_measure_change_basis():
    """Tests changing basis of a measurement.
    (Changes basis measurement to same basis as before.)
    """
    # bases to use
    old_basis = None
    new_basis = None

    # measure in first (old) basis
    measure = Measurement(2, range(2), old_basis)
    assert measure.num_qubits == 2
    assert measure.num_measurements == 2
    assert measure.measured_qubits == [0, 1]

    # change basis
    measure.change_basis(new_basis)
    assert measure.num_qubits == 2
    assert measure.num_measurements == 2
    assert measure.measured_qubits == [0, 1]


if __name__ == "__main__":
    test_measure_all()
    test_measure_single()
    test_measure_some()
    test_measure_tuple()
    test_measure_change_basis()
    print("All tests for Measurement passed.")
