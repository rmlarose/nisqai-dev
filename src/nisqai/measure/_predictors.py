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


"""Module containing pre-defined predictor functions. A predictor function inputs
a nisqai.measure.MeasurementOutcome and outputs a 0 or 1 according to some rules (defined
in the function). See examples.

Users can define their own predictor functions as long as they abide by these rules:
    (1) Inputs a single argument of type MeasurementOutcome.
    (2) Outputs a 0 or 1.
"""

# Imports
from numpy import argmax

from nisqai.measure import MeasurementOutcome


# Errors
class InvalidPredictor(Exception):
    pass


def _verify_measurement_outcome(meas):
    """Raises an InvalidPredictor if input is not of type
    nisqai.measure.MeasurementOutcome.
    """
    if type(meas) != MeasurementOutcome:
        raise InvalidPredictor("Argument to predictor must be of type MeasurementOutcome.")


def split_predictor(measurement_outcome):
    """Returns 0 if most measurements occur before the "halfway index" of
    all possible bit-strings that could be measured.

    Args:
        measurement_outcome : MeasurementOutcome
            The outcome of measuring a circuit.
    """
    # Check input
    _verify_measurement_outcome(measurement_outcome)

    # Compute the average
    average = measurement_outcome.average()

    # Edge case with only measuring one qubit
    if len(average) == 1:
        return 0 if average[0] <= 0.5 else 1

    # General case with measuring more than one qubit
    return 0 if argmax(average) < len(average) / 2 else 1
