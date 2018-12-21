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

"""Module for generating and working with product ansatzes."""

from numpy import empty, pi

from pyquil import gates

from _base_ansatz import BaseAnsatz


REAL_MEM_TYPE = "REAL"
BIT_MEM_TYPE = "BIT"


class ProductAnsatz(BaseAnsatz):
    """Class for working with product ansatze."""

    def __init__(self, num_qubits, gate_depth=3):
        BaseAnsatz.__init__(self, num_qubits)
        self.gate_depth = gate_depth
        self.write_circuit()

    def write_circuit(self):
        """Writes the product state ansatz circuit."""
        # empty list to store parameters
        # TODO: make this a Param class (see _params.py)
        self.params = empty((self.num_qubits, self.gate_depth), dtype=list)
        
        # 
        for q in range(self.num_qubits):
            for g in range(self.gate_depth):
                # make a parameter
                self.params[q, g] = self.circuit.declare(
                    (q, g), memory_type=REAL_MEM_TYPE)

        for q in range(self.num_qubits):
            for g in range(self.gate_depth):
                self.circuit.inst(
                    gates.RX(pi / 2, q),
                    gates.RZ(self.params[q, g], q)
                )