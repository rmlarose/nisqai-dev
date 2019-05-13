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

# TODO: write classical cost methods based on the output of the circuit and the data
# for example, indicator cost function based on measurement results as in the one-qubit-classifier

# should this class take in a "network" or "trainer" class?


def indicator(prediction, label):
    """Returns one if the prediction doesn't match the label, else zero.

    Args:

        prediction : int
            Prediction of a data point by a classifier.

        label : int
            Actual label of a data point by a classifier.
        """
    return 1 if prediction != label else 0

# TODO: implement
def cross_entropy(network_distribution, known_distribution):
    pass

# TODO: implement
def l2distance(network_distribution, known_distribution):
    pass