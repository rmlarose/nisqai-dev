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


"""Example of using NISQAI to visualize two-dimensional data."""

# Imports
import nisqai
import numpy as np

# Create some data
data = np.array([[0.1, 0.9], [0.3, 0.4], [0.5, 0.5], [0.2, 0.6]])

# Create a CData object
cdata = nisqai.data.CData(data)

# Visualize the data
nisqai.visual.scatter(cdata, color="black")

# Labels for each data point
labels = np.array([0, 1, 1, 0])

# Create a LabeledCData object
lcdata = nisqai.data.LabeledCData(data, labels)

# Color the data according to label and visualize it
nisqai.visual.scatter_with_labels(lcdata, color1="blue", color2="green")
