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
import matplotlib.pyplot as plt

from nisqai.data import CData, LabeledCData


# Errors and exceptions
class DimensionError(Exception):
    pass


def scatter(cdata, color="black"):
    """Shows a scatter plot for a CData (LabeledCData) object with two features."""
    # Make sure we have the correct input type
    if not isinstance(cdata, (CData, LabeledCData)):
        raise ValueError("Input type must be CData or LabeledCData.")

    # Make sure the data dimension is supported
    if cdata.num_features != 2:
        raise DimensionError(
            "Data must be two-dimensional. (Number of features must be two)."
        )

    # Scatter the data points
    for x in cdata.data:
        plt.scatter(x[0], x[1], color=color)

    # Plotting options
    plt.grid()

    # Show the plot
    plt.show()


def scatter_with_labels(cdata, color1="blue", color2="green"):
    """Shows a scatter plot for a LabeledCData object with two features,
    colored according to labels.

    Args:
        cdata : nisqai.data.LabeledCData
            Two-dimensional data to visualize.

        color1 : str
            Color to use for first class of data when plotting.

        color2: str
            Color to use for second class of data when plotting.
    """
    # Make sure we have the correct input type
    if not isinstance(cdata, LabeledCData):
        raise ValueError("cdata mst be of type LabeledCData.")

    # Make sure the data dimension is supported
    if cdata.num_features != 2:
        raise DimensionError(
            "Data must be two-dimensional. (Number of features must be two)."
        )

    # Scatter the data points
    for point, label in zip(cdata.data, cdata.labels):
        color = color1 if label == 0 else color2
        plt.scatter(point[0], point[1], color=color)

    # Plotting options
    plt.grid()

    # Show the plot
    plt.show()
