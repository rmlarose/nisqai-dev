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

from nisqai.layer._mera_ansatz import MeraAnsatz


def test_basic():
    """Tests that a ProductAnsatz can be instantiated."""
    # create a MERA ansatz on four qubits
    ansatz = MeraAnsatz(4)

    # check that the number of qubits is correct
    assert ansatz.num_qubits == 4


def test_params():
    """Tests the params attribute has the correct shape."""
    # create a MERA ansatz
    ansatz = MeraAnsatz(2**4)
    # test if the params attribute has the correct shape
    assert ansatz.params.shape() == (2**4, 2*4 - 1)


def test_correct_small():
    """Creates a small MeraAnsatz and tests if the circuit is correct."""
    # create a small MERA ansatz
    ansatz = MeraAnsatz(2**2)

    # correct string representation of program
    correct = "DECLARE q_000_g_000 REAL[1]\n" + \
            "DECLARE q_000_g_001 REAL[1]\n" + \
            "DECLARE q_000_g_002 REAL[1]\n" + \
            "DECLARE q_001_g_000 REAL[1]\n" + \
            "DECLARE q_001_g_001 REAL[1]\n" + \
            "DECLARE q_001_g_002 REAL[1]\n" + \
            "DECLARE q_002_g_000 REAL[1]\n" + \
            "DECLARE q_002_g_001 REAL[1]\n" + \
            "DECLARE q_002_g_002 REAL[1]\n" + \
            "DECLARE q_003_g_000 REAL[1]\n" + \
            "DECLARE q_003_g_001 REAL[1]\n" + \
            "DECLARE q_003_g_002 REAL[1]\n" + \
            "RY(q_001_g_000) 1\n" + \
            "RY(q_002_g_000) 2\n" + \
            "CNOT 1 2\n" + \
            "RY(q_000_g_001) 0\n" + \
            "RY(q_001_g_001) 1\n" + \
            "CNOT 0 1\n" + \
            "RY(q_002_g_001) 2\n" + \
            "RY(q_003_g_001) 3\n" + \
            "CNOT 2 3\n" + \
            "RY(q_001_g_002) 1\n" + \
            "RY(q_003_g_002) 3\n" + \
            "CNOT 1 3\n"
            

    # make sure the program is correct
    assert ansatz.circuit.__str__() == correct


if __name__ == "__main__":
    test_basic()
    test_params()
    test_correct_small()
    print("All tests for MeraAnsatz passed.")
