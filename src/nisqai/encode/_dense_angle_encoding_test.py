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

from nisqai.data._cdata import CData
from nisqai.encode._dense_angle_encoding import DenseAngleEncoding
from nisqai.encode._encoders import angle_simple_linear
from nisqai.encode._feature_maps import nearest_neighbor

from numpy import array


def test_simple():
    """Creates an AngleEncoding and performs simple checks."""
    data = array([[1, 2],
                  [3, 4]])
    cdata = CData(data)
    spreps = DenseAngleEncoding(cdata, encoder=angle_simple_linear, feature_map=nearest_neighbor(2, 1))
    assert len(spreps.circuits[0].circuit.instructions) == 0
    assert len(spreps.circuits[1].circuit.instructions) == 0
    spreps._write_circuit(0)
    print(spreps.circuits[0])
    spreps._write_circuit(1)
    print(spreps.circuits[1])


if __name__ == "__main__":
    test_simple()
    print("All tests for AngleEncoding passed.")
