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

from pyquil.gates import X

from nisqai.data._cdata import CData, LabeledCData
from nisqai.layer._base_ansatz import BaseAnsatz


class BinaryEncoding:
    """BinaryEncoding class. Writes classical binary data into a quantum state
    via a depth one circuit.

    |0>---[X^z_1]---
    |0>---[X^z_2]---
    ...
    |0>---[X^z_n]---

    Here, each z_i is a feature in the feature vector of length n.
    """
    def __init__(self, data):
        """Initializes a BinaryEncoding."""
        assert isinstance(data, (CData, LabeledCData))
        self.data = data

        # TODO: make sure the data consists of ints only
        # compute the number of qubits needed from the data
        num_qubits = self.data.num_features

        # store the circuits
        self.circuits = [BaseAnsatz(num_qubits) for _ in range(self.data.num_samples)]

        # write the circuits
        # TODO: utilize parametric compilation!
        for ind in range(self.data.num_samples):
            self._write_circuit(ind)

    def _write_circuit(self, feature_vector_index):
        """Writes the circuit for a particular index."""
        # grab the feature vector
        feature_vector = self.data.data[feature_vector_index]

        # compute the indices to put gates at
        inds = [x for x in range(len(feature_vector)) if feature_vector[x] == 1]

        # write the circuit
        self.circuits[feature_vector_index].add_at(X, inds)

    # TODO: all encoding classes will need this method.
    # TODO: make a BaseEncoding that implements this
    def __getitem__(self, ind):
        """Returns the circuit for the data point indexed by ind."""
        if not isinstance(ind, int):
            raise TypeError

        return self.circuits[ind]

    # TODO: all encoding classes will need this method.
    # TODO: make a BaseEncoding that implements this
    def __len__(self):
        """Returns the number of data points in the Encoder."""
        return self.data.num_samples
