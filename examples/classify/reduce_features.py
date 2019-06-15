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

"""Example NISQAI script demonstrating how to reduce features for the single qubit perceptron."""

# Imports
import numpy as np
import nisqai

# Some artificial data
mydata = np.array([
    [1, -1, 2, 1],
    [2, -1, -4, 5],
    [4, 3, 1, -2],
    [8, 6, 7, 5],
    [3, -1, 9, 9],
    [0, 1, 1, -0]
])

# Labels for the data
labels = np.array([1, 1, 0, 0, 1, 0])

# Create a LabeledCData object from the classical data
cdata = nisqai.data.LabeledCData(mydata, labels)

# Reduce the number of features
print(cdata.data)
cdata.reduce_features(0.5)
print(cdata.data)

