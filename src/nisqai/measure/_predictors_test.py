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

# Imports
import unittest
from numpy import array

from nisqai.measure._measurement_outcome import MeasurementOutcome
from nisqai.measure._predictors import split_predictor


class PredictorTest(unittest.TestCase):
    """Unit tests for predictor functions."""

    def test_split_predictor_zero(self):
        res = array([[1, 0], [1, 0], [1, 0], [1, 0]])
        meas = MeasurementOutcome(res)

        self.assertEqual(split_predictor(meas), 0)

    def test_split_predictor_one(self):
        res = array([[1, 1], [1, 1], [0, 1], [0, 0]])
        meas = MeasurementOutcome(res)

        self.assertEqual(split_predictor(meas), 1)


if __name__ == "__main__":
    unittest.main()
