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

from math import pi, log2

from pyquil import gates

from nisqai.layer._params import mera_ansatz_parameters
from nisqai.layer._base_ansatz import BaseAnsatz, REAL_MEM_TYPE
from nisqai.utils._program_utils import order


class MeraAnsatz(BaseAnsatz):
    """Class for working with MERA ansatze."""

    def __init__(self, num_qubits):
        """Initializes a MeraAnsatz.

        Args:
            num_qubits : int
                Number of qubits in the ansatz.

            gate_depth : int
                Number of alternating layers

                ----[Ry(parameter1)]----CNOT control----
                ----[Ry(parameter2)]----CNOT target-----

                This pattern is a gate_depth of one.
            """
        # initialize the BaseAnsatz class
        super().__init__(num_qubits)

        # store the gate depth
        self.gate_depth = int(log2(num_qubits))

        # get parameters for the ansatz
        self.params = mera_ansatz_parameters(
            self.num_qubits, self.gate_depth, 0.0
        )

        # declare all memory references to the ansatz circuit
        self.params.declare_memory_references(self.circuit)

        # write the circuit
        self._write_circuit()

    def _write_circuit(self):
        """Writes the MERA ansatz circuit."""
        depth = self.gate_depth
        for i in range(depth, 0, -1):
            for j in range(2):
                for g in range(2**(i-1) - 1 + j):
                    q = int(2**(depth-i+1)*(g-j/2+1)-1)
                    layer = 2*(depth-i)+j
                    if i == 1:
                        layer -= 1
                    # debugging
                    # print(self.params.memory_references[q][layer])
                    # print(self.params.memory_references[q + 2**(depth-i)][layer])
                    # print(q)
                    # print(q + 2**(depth-i))
                    # print(i)
                    # add gates into the circuit ansatz
                    self.circuit.inst(
                        gates.RY(self.params.memory_references[q][layer], q),
                        gates.RY(self.params.memory_references[q + 2**(depth-i)][layer], q + 2**(depth-i)),
                        gates.CNOT( q, q + 2**(depth-i) )
                    )
        self.circuit = order(self.circuit)
