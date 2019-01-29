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

from nisqai.layer._base_ansatz import BaseAnsatz
from pyquil.gates import MEASURE


class Measurement(BaseAnsatz):
    """Measurement class."""

    def __init__(self, num_qubits, qubits_to_measure, basis=None):
        """Initializes a Measurement.

        Args:
            qubits_to_measure : iterable
                Indices of qubits to perform measurements on.

            basis : string?
                Basis to measure qubits in.
        """
        # TODO: allow measurement in a different basis
        # input a string representing what basis? i.e., "Bell", etc.?
        # or input a list of gates representing what basis to measure in?
        super().__init__(num_qubits)
        self.creg = self.circuit.declare("ro", memory_size=len(qubits_to_measure))
        self.num_measurements = len(qubits_to_measure)
        self.measured_qubits = list(qubits_to_measure)
        self.circuit += [MEASURE(q, self.creg[ii]) for (ii, q) in enumerate(qubits_to_measure)]

    def change_basis(self, new_basis):
        """Changes the measurement basis to a new one."""
        return self.__init__(self.num_qubits, self.measured_qubits, basis=new_basis)
