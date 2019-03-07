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

from nisqai.utils._program_utils import order, ascii_circuit
from pyquil import Program, gates


def test_order_basic():
    """Tests an ordered program is in the nominal form."""
    p = Program(gates.H(0))
    p.declare("ro")
    print(order(p))


def test_ascii_circuit_1():
    """Do this"""
    print("I need to do this.")


if __name__ == "__main__":
    test_order_basic()
    test_ascii_circuit_1()
    print("All unit tests for program_utils passed.")
