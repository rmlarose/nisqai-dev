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
from pyquil.gates import MEASURE, H, RX, RZ, RY, CNOT, PHASE

class Measurement(BaseAnsatz):
    """Measurement class."""

    def __init__(self, num_qubits, qubits_to_measure, basis_gate=None, basis_angle=None):
        """Initializes a Measurement.

        Args:
            qubits_to_measure : iterable
                Indices of qubits to perform measurements on.

            basis_gate : string
                Basis to measure qubits in.

            basis_angle : float
                Rotation angle for parameterised single qubit gates.
        """
        if num_qubits <= 0:
            raise ValueError("Invalid number of qubits.")

        # TODO: allow measurement in a different basis
        # input a string representing what basis? i.e., "Bell", etc.?
        # or input a list of gates representing what basis to measure in?
        super().__init__(num_qubits)
        self.creg = self.circuit.declare("ro", memory_size=len(qubits_to_measure))
        self.num_measurements = len(qubits_to_measure)
        self.measured_qubits = list(qubits_to_measure)

        # all qubits measured in same basis e.g. ['RX', theta] measures in RX(theta)|0/1> basis.
        if basis_gate is not None:
            if basis_gate.lower()   == "rx":
                self.circuit += [RX(basis_angle, q) for q in qubits_to_measure]

            elif basis_gate.lower() ==  "ry":
                self.circuit += [RY(basis_angle, q) for q in qubits_to_measure]

            elif basis_gate.lower() ==  "rz":
                self.circuit += [RZ(basis_angle, q) for q in qubits_to_measure]

            elif basis_gate.lower() ==  "h":
                # Measurement in |+/-> basis
                self.circuit += [H(q) for q in qubits_to_measure]

            elif basis_gate.lower() ==  "xy":
                # Measurement in Pauli X-Y Plane, with angle : basis_angle.
                self.circuit += [PHASE(basis_angle, q) for q in qubits_to_measure]
                self.circuit += [H(q) for q in qubits_to_measure]

            elif basis_gate.lower() ==  "bell":
                # Bell basis measurement for 2 qubits.
                if num_qubits == 2:
                    self.circuit += CNOT(qubits_to_measure[0], qubits_to_measure[1])
                    self.circuit += H(qubits_to_measure[0])
                else: raise ValueError('Number of Qubits for Bell Measurement must be 2.')

            elif basis_gate.lower() ==  "ghz":
                # GHZ basis measurement for all qubits.
                self.circuit += [CNOT(qubits_to_measure[0], q) for q in qubits_to_measure[1:]]
                self.circuit += H(qubits_to_measure[0])
            
        self.circuit += [MEASURE(q, self.creg[ii]) for (ii, q) in enumerate(qubits_to_measure)]

    def change_basis(self, new_basis):
        """Changes the measurement basis to a new one."""
        if type(new_basis) is list:
            self.basis_gate = new_basis[0]
            self.basis_angle = new_basis[1]
        elif type(new_basis) is str:
            self.basis_gate = new_basis
            self.basis_angle = None
        else: raise TypeError("New basis must be either a list or a string")

        return self.__init__(self.num_qubits, self.measured_qubits, self.basis_gate, self.basis_angle)

    # TODO: write methods for getting output CData from a measurement result.
    # TODO: Example: forming a new CData object with data [prob(0), prob(1)] as features.


def measure_all(num_qubits):
    """Returns a Measurement on all qubits.

    Args:
        num_qubits : int
            Number of qubits in the circuit.
    """
    return Measurement(num_qubits, range(num_qubits))


def measure_top(num_qubits):
    """Returns a Measurement on only the first qubit.

    Args:
        num_qubits : int
            Number of qubits in the circuit.
    """
    return Measurement(num_qubits, [0])

def measure_qubit(num_qubits, qubit_index):
    """Returns a Measurement on only the qubit sepcified by qubit_index.

    Args:
        num_qubits : int
                    Number of qubits in the circuit.
        qubit_index : int
                    Qubit index on which measurement is performed
    """
    if qubit_index > num_qubits:
        raise ValueError("Qubit to be measured is not in available qubits")
    return Measurement(num_qubits, [qubit_index])