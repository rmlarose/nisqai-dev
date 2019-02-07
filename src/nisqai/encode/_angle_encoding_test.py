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

from numpy import array

from nisqai.data._cdata import CData
from nisqai.encode._angle_encoding import AngleEncoding
from nisqai.encode._encoders import angle
from nisqai.encode._feature_maps import direct


def test_simple():
    """Creates an AngleEncoding instance and performs simple checks."""
    data = array([[1], [2]])
    cdata = CData(data)
    angle_encoding = AngleEncoding(cdata, angle, direct(1))
    assert angle_encoding.data == cdata


def test_num_circuts():
    """Tests the number of circuits is equal to the number of samples."""
    data = array([[1], [2], [3], [4]])
    cdata = CData(data)
    angle_encoding = AngleEncoding(cdata, angle, direct(1))
    assert len(angle_encoding.circuits) == 4


if __name__ == "__main__":
    test_simple()
    test_num_circuts()
    print("All tests for AngleEncoding passed.")
