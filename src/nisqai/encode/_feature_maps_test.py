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


def test_direct_simple():
    """Tests the direct feature map with simple checks."""
    encoder = direct(10)
    assert isinstance(encoder, FeatureMap)
    for x in range(len(encoder.map)):
        assert len(encoder.map[x]) == 1
        assert x in encoder.map[x]


def test_nearest_neighbor_simple():
    """Performs simple tests on nearest neighbor feature map."""
    encoder = nearest_neighbor(4, 2)
    print(encoder.map)
    print(type(encoder.map))
    assert isinstance(encoder, FeatureMap)
    assert(encoder.map[0] == (0, 1))
    assert(encoder.map[1] == (2, 3))



if __name__ == "__main__":
    test_direct_simple()
    test_nearest_neighbor_simple()
    print("All tests for feature maps passed.")
