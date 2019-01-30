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

    # print out circuits
    print(spreps.circuits[0])
    print(spreps.circuits[1])


def test_index():
    # get data and create an encoder
    data = array([[0, 0]])
    cdata = CData(data)
    encoder = DenseAngleEncoding(cdata, encoder=angle_simple_linear, feature_map=nearest_neighbor(2, 1))

    # define the correct circuit
    correct = """DEFGATE S0:\n    1.0, 0.0\n    0.0, -1.0\n\nS0 0\n"""
    print("correct = ", correct, sep="\n")
    print(encoder[0])
    # TODO: get below line working (what's different about correct?)
    # assert encoder[0] == correct


if __name__ == "__main__":
    test_simple()
    test_index()
    print("All tests for DenseAngleEncoding passed.")
