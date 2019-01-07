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

from nisqai.encode._feature_maps import (FeatureMap,
                                         direct,
                                         nearest_neighbor)
from nisqai.data._cdata import CData

from numpy import array, empty


def test_direct_simple():
    """Tests the direct feature map with simple checks."""
    feature_map = direct(10)
    assert isinstance(feature_map, FeatureMap)
    for x in range(len(feature_map.map)):
        assert len(feature_map.map[x]) == 1
        assert x in feature_map.map[x]


def test_nearest_neighbor_simple():
    """Performs simple tests on nearest neighbor feature map."""
    feature_map = nearest_neighbor(4, 2)
    print(feature_map.map)
    print(type(feature_map.map))
    assert isinstance(feature_map, FeatureMap)
    assert(feature_map.map[0] == (0, 1))
    assert(feature_map.map[1] == (2, 3))


def test_nearest_neighbor_odd():
    """Tests the nearest neighbor feature map with an odd number of features."""
    pass


def test_nearest_neighbor_data_features_to_qubits_map():
    """"""
    # example data
    data = array([[1, 2, 3, 4],
                  [2, 1, 3, 4]])
    cdata = CData(data)
    feature_vector = cdata.data[0]

    # nearest neighbor feature map
    feature_map = nearest_neighbor(4, 2)
    print(feature_map.map)
    # test the encoding
    qubit_features = {}
    for ind in range(len(feature_map.map)):
        print(ind)
        print([x for x in feature_map.map[ind]])
        features = []
        for x in feature_map.map[ind]:
            features.append(feature_vector[x])
        qubit_features[ind] = features

if __name__ == "__main__":
    test_direct_simple()
    test_nearest_neighbor_simple()
    test_nearest_neighbor_data_features_to_qubits_map()
    print("All tests for feature maps passed.")
