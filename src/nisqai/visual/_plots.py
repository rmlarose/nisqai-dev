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
from numpy import ndarray
import matplotlib.pyplot as plt

from nisqai.data import CData, LabeledCData


# Errors and exceptions
class DimensionError(Exception):
    pass


def scatter(data, labels=None, color1="blue", color2="green"):
    """Shows a scatter plot for two-dimensional data. Points are colored by label if labels are provided.

    Args:
        data : numpy.ndarray
            Two-dimensional data to visualize. Each row should be a data point, and the number of columns is the
            total number of data points.

        labels : array-like
            Array of labels/predictions (nominally valued 0 or 1). Must be of the same linear dimension as data.

        color1 : str
            Color to use for first class of data when plotting.

        color2: str
            Color to use for second class of data when plotting.
    """
    # Make sure we have the correct input type
    if not isinstance(data, ndarray):
        raise ValueError("data must be of type numpy.ndarray.")

    # Get the shape of the data
    num_points, num_features = data.shape

    # Make sure the data dimension is supported
    if num_features != 2:
        raise DimensionError(
            "Data must be two-dimensional. (Number of features must be two)."
        )

    # If labels are provided
    if labels is not None:
        # Get the unique labels
        if 1 > len(set(labels)) > 2:
            raise ValueError("Invalid number of labels. There should be one or two unique labels.")

        # Get the unique labels
        if len(set(labels)) == 1:
            label1 = set(labels)
        else:
            label1, _ = set(labels)

        # Scatter the data points
        for point, label in zip(data, labels):
            color = color1 if label == label1 else color2
            plt.scatter(point[0], point[1], color=color)

    # If labels are not provided
    else:
        for point in data:
            # Scatter the points
            plt.scatter(point[0], point[1], color=color1)

    # Plotting options
    plt.grid()

    # Show the plot
    plt.show()
