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

from nisqai.layer._alternating_ansatz import AlternatingAnsatz
from pyquil import Program, gates


def test_basic():
    """Makes sure an AlternatingAnsatz can be created and
    has the correct number of qubits.
    """
    a = AlternatingAnsatz(2)
    assert a.num_qubits == 2


def test_match_expected_program():
    """Tests the output of an AlternatingAnsatz against a given
    correct output.
    """
    a = AlternatingAnsatz(3, structure=[gates.RX])

    # make a program with the correct output
    p = Program()
    p00 = p.declare((0, 0), memory_type="REAL")
    p01 = p.declare((0, 1), memory_type="REAL")
    p10 = p.declare((1, 0), memory_type="REAL")
    p11 = p.declare((1, 1), memory_type="REAL")
    p20 = p.declare((2, 0), memory_type="REAL")
    p21 = p.declare((2, 1), memory_type="REAL")
    p.inst(
        gates.RX(p00, 0),
        gates.RX(p10, 1),
        gates.RX(p20, 2),
        gates.CZ(0, 1),
        gates.RX(p01, 0),
        gates.RX(p11, 1),
        gates.RX(p21, 2),
        gates.CZ(1, 2)
    )

    assert a.circuit == p


def test_odd_qubits():
    """Tests the AlternatingAnsatz circuit is correct for an
    odd number of qubits.
    """
    a = AlternatingAnsatz(3)
    print(a)


def test_structure():
    """Provides a different structure to an AlternatingAnsatz
    and asserts the output is correct."""
    a = AlternatingAnsatz(2, [gates.RY])
    assert a.structure == [gates.RY]
    print(a)


# run the tests
if __name__ == "__main__":
    test_basic()
    test_match_expected_program()
    test_odd_qubits()
    test_structure()
    print("All tests for AlternatingAnsatz passed.")
