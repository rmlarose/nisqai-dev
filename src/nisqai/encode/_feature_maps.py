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


class FeatureMap():
    """FeatureMap class."""

    def __init__(self, mapping):
        """Initialize a FeatureMap.

        Args:
            mapping : dict
                Dictionary of (qubit, features for qubit) key-value pairs.
        """
        self.map = mapping

    def _has_all_features(self):
        """Checks to make sure all features are present."""
        # TODO: implement
        pass

    def _is_valid_mapping(self):
        """Returns True if the mapping is valid."""
        # TODO: implement
        # TODO: what defines a valid mapping?
        pass


def direct(num_features):
    """Returns a FeatureMap with the direct encoding

    Feature[i] --> Qubit[i].
    """
    mapping = dict((k, (k,)) for k in range(num_features))
    return FeatureMap(mapping)


def nearest_neighbor(num_features, num_qubits):
    """Returns a FeatureMap with nearest neighbor encoding.

    Examples:
        nearest_neighbor(4, 2) --> {0 : (0, 1), 1 : (2, 3)}
    """
    bin_size = num_features // num_qubits
    mapping = dict((k, tuple(range(k * bin_size, (k + 1) * bin_size))) for k in range(0, num_qubits))
    return FeatureMap(mapping)


def group_biggest(data, num_features, num_qubits):
    """Returns a FeatureMap with the biggest features in the
    first qubits.

    Examples:
    """


def group_smallest(data, num_features, num_qubits):
    """Returns a FeatureMap with smallest features
    in the first qubits.

    Examples:
    """
