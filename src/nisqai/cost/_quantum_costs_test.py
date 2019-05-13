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

from pyquil.gates import X

from nisqai.cost import HilbertSchmidtDistance
from nisqai.layer._base_ansatz import BaseAnsatz
from nisqai.utils._program_utils import ascii_drawer


def test_basic():
    """Visual test of HilbertSchmidtDistance."""
    # number of qubits
    n = 2

    # example circuit
    ansatz = BaseAnsatz(n)

    # adding arbitrary gates
    ansatz.circuit += [X(q) for q in range(n)]

    # form the HST circuit
    hstansatz = HilbertSchmidtDistance(ansatz)

    # display the circuit
    print(hstansatz.hst.circuit)


if __name__ == "__main__":
    test_basic()
