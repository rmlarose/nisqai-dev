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
import time
import nisqai
import numpy as np

# Start the Rigetti QVM and Quil compiler
engine = nisqai.utils.startQVMandQUILC()

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
print("Data before PCA:")
print("Number of features:", cdata.num_features)
print(cdata.data)

cdata.reduce_features(0.5)

print("\nData after PCA:")
print("Number of features:", cdata.num_features)
print(cdata.data)

# Scale the features
print("\nData after scaling:")
cdata.scale_features("L1 norm")
print(cdata.data)

# Use a dense angle encoding (two features per qubit)
encoder = nisqai.encode.DenseAngleEncoding(
    cdata, nisqai.encode.angle_simple_linear, nisqai.encode.nearest_neighbor(2, 1)
)

# Define the ansatz for the network
ansatz = nisqai.layer.ProductAnsatz(1)

# Define the measurement scheme for the network
measure = nisqai.measure.Measurement(1, [0])

# Create a network
qnn = nisqai.network.Network([encoder, ansatz, measure], "1q-qvm", predictor=nisqai.measure.split_predictor)

# Train the network
start = time.time()
res = qnn.train(trainer="COBYLA", initial_angles=[0, 0, 0], shots=1000)

# Print the output
print("Train time:", (time.time() - start) / 60, "minutes.")
print("Train result:")
print(res)

# Stop the Rigetti QVM and Quil compiler
engine.stopQVM()
engine.stopQUILC()
