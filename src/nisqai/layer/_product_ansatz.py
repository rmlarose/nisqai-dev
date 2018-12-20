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

from _base_ansatz import BaseAnsatz

class ProductAnsatz(BaseAnsatz):
    """Class for working with product ansatze."""

    def __init__(self, num_qubits, gate_depth, angles):
        BaseAnsatz.__init__(self, num_qubits)
        self.gate_depth = gate_depth
        self.init_angles = angles
        # TODO: make classes for Parameters and GateAlphabets

    def write_circuit(self, angles):
        """Writes the product state ansatz circuit."""
        # TODO: complete method
        # self.circuit = 
        pass