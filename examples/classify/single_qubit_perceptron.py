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

"""Example NISQAI script demonstrating the single qubit perceptron on random data."""

# Imports
import time

import nisqai

# Start the Rigetti QVM and Quil compiler
qvm_server, quilc_server, _ = nisqai.utils.startQVMandQUILC()

# Get random two dimensional data
cdata = nisqai.data.random_data_vertical_boundary(50)

# Visualize the data
nisqai.visual.scatter(cdata.data, cdata.labels)

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
res = qnn.train(trainer="COBYLA", initial_angles=[0, 0, 0], updates=True, shots=1000)

# Print the output
print("Train time:", (time.time() - start) / 60, "minutes.")
print("Train result:")
print(res)

# Get the predictions at the optimal angles
predictions = qnn.predict_all(angles=res["x"])

# Visualize the classified data
nisqai.visual.scatter(cdata.data, labels=cdata.labels, predictions=predictions)

# Stop the Rigetti QVM and Quil compiler
nisqai.utils.stopQVMandQUILC(qvm_server, quilc_server)
