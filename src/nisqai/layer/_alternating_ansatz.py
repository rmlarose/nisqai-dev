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

from _base_ansatz import BaseAnsatz, REAL_MEM_TYPE

class AlternatingAnsatz(BaseAnsatz):
    """Class for two-qubit alternating ansatz."""
    
    def __init__(self, num_qubits, structure=None):
        super().__init__(num_qubits)
        self.structure = structure

    def write_circuit(self):
        """Adds instructions to the circuit."""
        pass