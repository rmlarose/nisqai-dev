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

from nisqai.data._cdata import CData, LabeledCData, random_data
from numpy import array
from numpy.random import rand


def test_basic_cdata():
    """Creates a CData object and makes sure the dimensions are correct."""
    data = array([[1, 0, 0], [0, 1, 0]])
    cdata = CData(data)
    assert cdata.num_features == 3
    assert cdata.num_samples == 2


def test_basic_labeled_cdata():
    """Creates a LabeledCData object and makes sure the dimensions are correct."""
    data = array([[1, 0, 0], [0, 1, 0]])
    labels = array([1, 0])
    lcdata = LabeledCData(data, labels)
    assert lcdata.num_features == 3
    assert lcdata.num_samples == 2


def test_get_random_data_basic():
    """Tests to see if we can get random data."""
    cdata = random_data(num_features=2,
                        num_samples=4,
                        labels=None)
    assert cdata.num_features == 2
    assert cdata.num_samples == 4


if __name__ == "__main__":
    test_basic_cdata()
    test_basic_labeled_cdata()
    test_get_random_data_basic()
    print("All tests for CData passed.")