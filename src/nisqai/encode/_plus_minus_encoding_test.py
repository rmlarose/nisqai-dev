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

from nisqai.encode._plus_minus_encoding import PlusMinusEncoding
from nisqai.data._cdata import CData


def test_basic():
    """Tests that a PlusMinusEncoding can be instantiated."""
    data = array([[1, 0, 0, 1],
                  [0, 1, 1, 0]], dtype=int)
    cdata = CData(data)
    encoder = PlusMinusEncoding(cdata)

    assert encoder.data == cdata
    assert len(encoder.circuits) == 2
    assert len(encoder) == 2


def test_correct():
    """Tests for correctness in the circuit of a PlusMinusEncoding, general case."""
    data = array([[1, 0, 0, 1]], dtype=int)
    cdata = CData(data)
    encoder = PlusMinusEncoding(cdata)

    correct = "H 0\n" + \
              "H 1\n" + \
              "H 2\n" + \
              "H 3\n" + \
              "Z 0\n" + \
              "Z 3\n"

    assert encoder[0].__str__() == correct


def test_correct_edge():
    """Tests for correctness in the circuit of a PlusMinusEncoding, edge case."""
    data = array([[0, 0, 0, 0]], dtype=int)
    cdata = CData(data)
    encoder = PlusMinusEncoding(cdata)

    correct = "H 0\n" + \
              "H 1\n" + \
              "H 2\n" + \
              "H 3\n"

    assert encoder[0].__str__() == correct


def test_correct_edge2():
    """Tests for correctness in the circuit of a PlusMinusEncoding, edge case."""
    data = array([[1, 1, 1, 1]], dtype=int)
    cdata = CData(data)
    encoder = PlusMinusEncoding(cdata)

    correct = "H 0\n" + \
              "H 1\n" + \
              "H 2\n" + \
              "H 3\n" + \
              "Z 0\n" + \
              "Z 1\n" + \
              "Z 2\n" + \
              "Z 3\n"

    assert encoder[0].__str__() == correct


if __name__ == "__main__":
    test_basic()
    test_correct()
    test_correct_edge()
    test_correct_edge2()
