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
from nisqai.data._cdata import CData, LabeledCData

from pyquil import Program

from numpy import array, cos, sin, isclose, dot, identity


class AngleEncoding():
    """AngleEncoding class."""

    def __init__(self, data, encoder, feature_map):
        """Inititiate an AngleEncoding class."""
        # TODO: better error checking
        assert isinstance(data, (CData, LabeledCData))
        self.data = data

        self.encoder = encoder
        self.feature_map = feature_map

        # determine number of qubits
        num_qubits = self._compute_num_qubits()

        self.circuits = [BaseAnsatz(num_qubits) for _ in range(self.data.num_samples)]

    def _compute_num_qubits(self):
        """Computes the number of qubits needed for the circuit
        from the input data.
        """
        # TODO: write in terms of the encoder
        return self.data.num_features

# TODO: make the angles parameters and store one circuit, instantiating the parameters
    def _write_circuit(self, feature_vector_index):
        """Writes the encoding circuit into self.circuit."""
        # grab the feature vector to create a circuit with
        feature_vector = self.data.data[feature_vector_index]

        # program to write
        prog = Program()

        # ===============================
        # collect features for each qubit
        # ===============================
        # example: for direct with linear encoding
        # qubit_features[0] = [feature_vector[0]]
        # qubit_features[1] = [feature_vector[1]]
        # etc.

        qubit_features = {}
        for ind in range(len(self.feature_map.map)):
            # temp list for all features
            features = []
            for x in self.feature_map.map[ind]:
                features.append(feature_vector[x])
            qubit_features[ind] = features

        # ==============================================================
        # use the encoder to get angles from the features for each qubit
        # ==============================================================

        qubit_angles = {}
        for (qubit_index, feature) in qubit_features.items():
            qubit_angles[qubit_index] = self.encoder(feature)

        # ============================
        # get matrices from the angles
        # ============================

        qubit_state_preps = {}
        for (qubit_index, angles) in qubit_angles.items():
            qubit_state_preps[qubit_index] = angle_to_matrix(angles)

        # ==================================
        # use each matrix to write a circuit
        # ==================================

        for (qubit_index, mat) in qubit_state_preps.items():
            # define the gate
            name = "S" + str(qubit_index)
            prog.defgate(name, mat)
            # write the gate into the circuit
            prog += (name, qubit_index)

        # write the program into the circuit of the ansatz
        self.circuits[feature_vector_index].circuit = prog


def angle_to_matrix(theta):
    """Converts a an angle into a state preparation matrix
    preparing the state encoding the angle

    cos(angle) |0> + sin(angle) |1>

    from the ground state."""
    # TODO: check that angle is within the right range

    # form the matrix
    mat = array([[cos(theta), sin(theta)],
                 [sin(theta), -1 * cos(theta)]])

    # TODO: better error checking
    assert isclose(dot(mat, mat.conj().T), identity(mat.shape[0])).all()

    return mat
