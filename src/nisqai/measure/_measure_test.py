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


def test_measure_simple():
    """Checks the ability to instantiate a measurement."""
    measure = Measurement(9, range(9))
    assert measure.num_qubits == 9
    assert measure.num_measurements == 9
    assert len(measure.circuit.instructions) == 10


def test_measure_single():
    measure = Measurement(10, [1])
    assert measure.num_qubits == 10
    assert measure.num_measurements == 1
    assert len(measure.circuit.instructions) == 2


if __name__ == "__main__":
    test_measure_simple()
    test_measure_single()
    print("All tests for Measurement passed.")
