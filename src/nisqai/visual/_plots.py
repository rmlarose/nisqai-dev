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


def scatter(data, labels=None, predictions=None, **kwargs):
    """Shows a scatter plot for two-dimensional data. Points are colored by label if labels are provided.

    Args:
        data : numpy.ndarray
            Two-dimensional data to visualize. Each row should be a data point, and the number of columns is the
            total number of data points.

        labels : array-like
            Array of labels (nominally valued 0 or 1). Must be of the same linear dimension as data.

        predictions : array-like
            Array of predictions (nominally valued 0 or 1). Must be of the same linear dimension as data.

        color1 : str
            Color to use for first class of data when plotting.

        color2: str
            Color to use for second class of data when plotting.
    """
    # Constants
    COLORS = ["blue", "orange", "green", "salmon", "red", "black", "purple"]
    ALPHA = 0.6
    SIZE = 75
    LINEWIDTH = 2

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

    # If we just get predictions and no labels, color the plot as if the predictions were labels
    if labels is None and predictions is not None:
        labels = predictions
        predictions = None

    # If labels are provided
    if labels is not None:
        # Make sure there is at least one unique label
        if len(set(labels)) < 1:
            raise ValueError("Invalid number of labels. There should be at least one unique label.")

        # Get the unique labels
        unique_labels = list(set(labels))
        if len(unique_labels) > len(COLORS):
            RuntimeWarning(
                "There are too many classes for supported colors. Duplicate colors will be used." +
                "To avoid this, pass in a list of string colors to scatter as a kwarg."
            )
            COLORS = COLORS * len(unique_labels)
        color_for_label = dict(zip(unique_labels, COLORS[:len(unique_labels)]))

        # Scatter the data points with labels but no predictions
        if predictions is None:
            for point, label in zip(data, labels):
                plt.scatter(point[0], point[1], color=color_for_label[label], s=SIZE, alpha=ALPHA)

        # Scatter the data points with labels and predictions
        else:
            for point, label, prediction in zip(data, labels, predictions):
                plt.scatter(
                    point[0], point[1], color=color_for_label[label], edgecolor=color_for_label[prediction],
                    linewidth=LINEWIDTH, s=SIZE, alpha=ALPHA
                )

    # If neither labels nor predictions are not provided
    else:
        for point in data:
            # Scatter the points
            plt.scatter(point[0], point[1], s=SIZE, alpha=ALPHA)

    # Put the score on the title
    if predictions is not None and labels is not None:
        num_wrong = sum(abs(labels - predictions))
        percent_correct = 100.0 - num_wrong / len(labels) * 100.0
        plt.title("Score: %0.3f" % percent_correct + "%")

    # Show the plot
    plt.show()
