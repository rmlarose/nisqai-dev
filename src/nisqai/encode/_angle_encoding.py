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
from nisqai.data._cdata import CData

from numpy import array, cos, sin, exp
from pyquil import Program


class AngleEncoding(BaseAnsatz):
    """AngleEncoding class. Encode features into the angles of qubits via

    |\psi> = cos(\theta/2) |0> + e^{i \phi} sin(\theta / 2) |1>.


    With this encoding, n features require ceiling(n / 2) qubits.

    Args:

        num_qubits : int
            The number of qubits in the circuit.

        data : nisqai.data.CData or nisqai.data.LabeledCData
            Data object to be encoded in the circuit.

        encoder : callable
            Function of the features...

        feature_map : callable or iterable
            Defines which features get encoded in which qubits.

            The first two features are mapped to the first qubit, the next two
            features are mapped to the second qubit, etc., unless a feature map
            is given, in which case features are mapped to qubit as specified by
            the feature map.
    """

    def __init__(self, data, encoder, feature_map):
        """Initialize an AngleEncoding class."""
        # TODO: replace with better error checking
        assert isinstance(CData, data)
        self.data = data

        # determine the number of qubits from the input data
        num_qubits = self._compute_num_qubits()
        super().__init__(self, num_qubits)
        self.encoder = encoder
        self.feature_map = feature_map

    def _compute_num_qubits(self):
        """Computes the number of qubits needed for the circuit
        from the input data.
        """
        return self.data.num_features // 2 + self.data.num_features % 2

    def _write_circuit(self):
        """Writes the encoding circuit into self.circuit."""
        # ===============================
        # collect features for each qubit
        # ===============================
        # example: for nearest_neighbor with linear encoding
        # qubit_features[0] = (data[0], data[1])
        qubit_features = []
        for ind in range(len(self.feature_map)):
            qubit_features[ind] = (self.data[x] for x in self.feature_map[ind])

        # ==============================================================
        # use the encoder to get angles from the features for each qubit
        # ==============================================================

        # ============================
        # get matrices from the angles
        # ============================

        # ==================================
        # use each matrix to write a circuit
        # ==================================

