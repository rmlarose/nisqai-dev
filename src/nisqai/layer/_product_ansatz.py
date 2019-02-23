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

from math import pi

from pyquil import gates

from nisqai.layer._params import product_ansatz_parameters
from nisqai.layer._base_ansatz import BaseAnsatz, REAL_MEM_TYPE
from nisqai.utils._program_utils import order


class ProductAnsatz(BaseAnsatz):
    """Class for working with product ansatze."""

    def __init__(self, num_qubits, gate_depth=3):
        """Initializes a ProductAnsatz.

        Args:
            num_qubits : int
                Number of qubits in the ansatz.

            gate_depth : int
                Number of hardware gates in the pattern

                ----[Rx(pi / 2)]----[Rz(parameter)]----

                This pattern is a gate_depth of one.
            """
        # initialize the BaseAnsatz class
        super().__init__(num_qubits)

        # store the gate depth
        self.gate_depth = gate_depth

        # get parameters for the ansatz
        self.params = product_ansatz_parameters(
            self.num_qubits, self.gate_depth, 0.0
        )

        # declare all memory references to the ansatz circuit
        self.params.declare_memory_references(self.circuit)

        # write the circuit
        self._write_circuit()

    def _write_circuit(self):
        """Writes the product state ansatz circuit."""
        for q in range(self.num_qubits):
            for g in range(self.gate_depth):
                # add gates into the circuit ansatz
                self.circuit.inst(
                    gates.RX(pi / 2, q),
                    gates.RZ(self.params.memory_references[q][g], q)
                )
        self.circuit = order(self.circuit)

    # TODO: make sure __sum__ works as intended for ProductAnsatz class
    # TODO: (see note in BaseAnsatz.__sum__())
