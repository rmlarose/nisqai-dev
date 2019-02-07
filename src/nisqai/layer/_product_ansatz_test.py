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

from nisqai.layer._product_ansatz import ProductAnsatz


def test_basic():
    """Tests that a ProductAnsatz can be instantiated."""
    # create an product ansatz on four qubits
    ansatz = ProductAnsatz(4)

    # check that the number of qubits is correct
    assert ansatz.num_qubits == 4


def test_params():
    """Tests the params attribute has the correct shape."""
    # create a product ansatz
    ansatz = ProductAnsatz(5, gate_depth=4)

    # test if the params attribute has the correct shape
    assert ansatz.params.shape == (5, 4)


def test_correct_small():
    """Creates a small ProductAnsatz and tests if the circuit is correct."""
    # create a small product ansatz
    ansatz = ProductAnsatz(1)

    # correct string representation of program
    correct = "DECLARE (0, 0) REAL[1]\nDECLARE (0, 1) REAL[1]\n" + \
              "DECLARE (0, 2) REAL[1]\nRX(pi/2) 0\nRZ((0, 0)) 0\n" + \
              "RX(pi/2) 0\nRZ((0, 1)) 0\nRX(pi/2) 0\nRZ((0, 2)) 0\n"

    # make sure the program is correct
    assert ansatz.circuit.__str__() == correct


if __name__ == "__main__":
    test_basic()
    test_params()
    test_correct_small()
    print("All tests for ProductAnsatz passed.")
