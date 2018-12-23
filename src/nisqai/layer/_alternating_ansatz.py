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
from pyquil import gates
from numpy import empty

class AlternatingAnsatz(BaseAnsatz):
    """Class for two-qubit alternating ansatz."""
    
    def __init__(self, num_qubits, structure=[gates.RZ, gates.RX, gates.RZ]):
        super().__init__(num_qubits)
        self.structure = structure
        self.depth = 2 * len(structure)
        self._make_params()
        self.write_circuit()

    def _make_params(self):
        """Adds a class attribute with all parameters needed for the ansatz."""
        self.params = empty((self.num_qubits, self.depth), dtype=list)
        for q in range(self.num_qubits):
            for g in range(self.depth):
                # make a parameter
                self.params[q, g] = self.circuit.declare(
                    (q, g), memory_type=REAL_MEM_TYPE)

    def write_circuit(self):
        """Adds instructions to the circuit."""
        # TODO: speedup, combine multiple loops into one
        # there must be a method to insert gates between other gates in pyquil
        # this would mean we could add all rotations at once and therefore
        # eliminate a loop over n
        # for brevity
        n = self.num_qubits

        # first layer of rotations on all qubits
        for q in range(n):
            self._rot(q, 0)

        # first layer of CNOTs
        stop = n - (n % 2)
        self.circuit.inst(
            [gates.CZ(q, (q + 1) % n) for q in range(0, stop, 2)]
            )

        # second layer of rotations on all qubits
        for q in range(n):
            self._rot(q, 1)

        # second layer of CNOTs
        stop = n - (n % 2)
        self.circuit.inst(
            [gates.CZ(q, (q + 1) % n) for q in range(1, stop, 2)]
            )

    def _two_qubit_block(self, qubit_index):
        """Writes a "two qubit block" on qubits labeled by qubit_index and
        qubit_index + 1 into the circuit.
        
        The "two-qubit block" has the following structure:
        
        ----[R1]----@----[R2]----@----
                    |            |
        ----[R3]----X----[R4]----X----
        
        where each R1, ..., R4 is an arbitrary sequence of single qubit
        rotations.
        """
        pass

    def _rot(self, qubit, num):
        """Adds a rotation to the qubit.
        
        Args:
            qubit [type: int]
                Index of qubit to be rotated.

            num [type: int]
                The first or second rotation on the qubit.
                Needed for indexing parameters.
        
        Modifies:
            self.circuit
        """
        for (g, gate) in enumerate(self.structure):
            self.circuit.inst(
                gate(self.params[qubit, g + (self.depth // 2) * num], qubit)
                )