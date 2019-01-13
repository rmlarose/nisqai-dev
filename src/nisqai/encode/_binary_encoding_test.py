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

from nisqai.encode._binary_encoding import BinaryEncoding
from nisqai.data import CData
from numpy import array


def test_construct():
    """Test ability to instantiate a BinaryEncoding."""
    data = array([[1, 0, 0, 1],
                  [0, 1, 1, 0],
                  [1, 1, 0, 1],
                  [0, 1, 1, 1]], dtype=int)
    cdata = CData(data)
    encoding = BinaryEncoding(cdata)
    assert len(encoding.circuits) == 4


def test_circuits():
    """Tests for correctness of the state prep circuits."""
    data = array([[1, 0, 0, 1],
                  [0, 1, 1, 0],
                  [1, 1, 0, 1],
                  [0, 1, 1, 1]], dtype=int)
    cdata = CData(data)
    encoding = BinaryEncoding(cdata)
    print(encoding.circuits[2])

if __name__ == "__main__":
    test_construct()
    test_circuits()
    print("All tests for BinaryEncoding passed.")
