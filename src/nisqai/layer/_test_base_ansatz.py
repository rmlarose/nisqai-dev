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


def test_basic():
    """Basic test for BaseAnsatz."""
    b = BaseAnsatz(4)
    assert b.num_qubits == 4


def test_add():
    """Add two BaseAnsatz's and make sure the result is correct."""
    a = BaseAnsatz(3)
    b = BaseAnsatz(3)
    
    a.add_layer(gates.X)
    b.add_layer(gates.Y)
    
    print("a + b =\n", a + b)


if __name__ == "__main__":
    test_basic()
    test_add()
    print("All tests for BaseAnsatz class passed.")
