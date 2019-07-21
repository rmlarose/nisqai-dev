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

from math import ceil
import os

from copy import deepcopy
import numpy as np

from torchvision import datasets

from nisqai.data.data_sets import iris


class CData:
    """Classical data class."""

    def __init__(self, data):
        """Initialize a CData object.

        Args:
            data [type: numpy array]
                data values, shape should be (samples, features).
        """
        # TODO: allow data to be a pandas dataframe, 2d list, and
        #  other possible data types people would just want to throw
        #  into the class without thinking about it
        #  for pandas dataframes, just need to convert it to an array
        self.raw_data = deepcopy(data)
        self.data = deepcopy(self.raw_data)

        # Descriptors for the data set
        self._centered = False
        self._centered = self.is_centered()

    @property
    def num_features(self):
        """Returns the number of features in the data set.

        Return type: int
        """
        return self.data.shape[1]

    @property
    def num_samples(self):
        """Returns the number of samples (feature vectors) in the data set.

        Return type: int
        """
        return self.data.shape[0]

    def mean(self):
        """Returns the mean of the data."""
        return np.mean(self.data, axis=0)

    def center(self):
        """Modifies data by subtracting the mean."""
        if not self._centered:
            self.data = self.data - np.mean(self.data, axis=0)
            self._centered = True

    def is_centered(self, tolerance=1e-3):
        """Returns True if the data set is centered, else False."""
        # If we know the data is already centered, return True
        if self._centered:
            return True

        # Otherwise, do the standard check
        if abs(sum(self.mean()) - 0.0) > tolerance:
            self._centered = False
        else:
            self._centered = True
        return self._centered

    def scale_features(self, method):
        """Modifies features of data by scaling them according to a specified method.

        Args:
            method : str
                Key string which specifies the desired feature scaling method.
                Options:

                    'min-max norm'
                        x' = (x - min(x))/(max(x) - min(x))

                    'mean norm'
                        x' = (x-avg(x))/(max(x) - min(x))

                    'standardize'
                        x' = (x - avg(x))/sigma

                    'L2 norm'
                        x' = x / L2norm(x)

                    'L1 norm'
                        x' = x / L1norm(x)

                    "inf norm"
                        Divides each feature by the max value for that feature over all samples.

        Returns:
            None

        Modifies:
            self.data
        """
        # Try to catch wrong string formatting
        method = method.lower().strip()

        # Min-max norm
        if method == 'min-max norm':
            mmin = self.data.min(axis=0)
            mmax = self.data.max(axis=0)
            self.data = (self.data - mmin) / (mmax - mmin)

        # Mean norm
        elif method == 'mean norm':
            mean = self.data.mean(axis=0)
            mmin = self.data.min(axis=0)
            mmax = self.data.max(axis=0)
            self.data = (self.data - mean) / (mmax - mmin)

        # Standardize
        elif method == 'standardize':
            mean = self.data.mean(axis=0)
            # ddof = 1 gives sample sd
            sd = self.data.std(axis=0, ddof=1)
            self.data = (self.data - mean) / sd

        # L2 norm
        elif method == 'L2 norm' or method == "l2 norm":
            norm = np.linalg.norm(self.data, ord=2, axis=0)
            self.data = self.data / norm

        # L1 norm
        elif method == 'L1 norm' or method == "l1 norm":
            L1norm = sum(abs(self.data))
            self.data = self.data / L1norm

        # Infinity norm
        elif method in ("inf norm", "infty norm", "infinity norm", "inf", "infty", "infinity"):
            self.data /= np.max(self.data, axis=0)

        else:
            raise ValueError("Unsupported scaling method. See help(scale_features) for supported methods.")

    def reduce_features(self, fraction):
        """Performs (classical) principal component analysis
         on the data and keeps the desired fraction of features.

         Modifies self.data in place.

        Args:
            fraction [type: float]
                Keeps this ratio of features.

        Example:
            reduce_features(0.2) --> keeps top 20% of features after PCA
        """
        # Center columns by subtracting the column mean
        if not self._centered:
            self.center()

        # Calculate the covariance of centered matrix
        # Note: np.cov expects each row to be variable (i.e. feature)
        covariance = np.cov(self.data.T)

        # Get eigenvectors of covariance matrix
        _, evecs = np.linalg.eig(covariance)

        # Project data with first column as first principle component
        projected = evecs.T.dot(self.data.T).T

        # Only keep the input fraction of features
        nfeatures = ceil(fraction * self.num_features)

        self.data = projected.T[:nfeatures].T

    def pad_one(self):
        """Appends a zero element to each data point, increasing the dimension by one.

        Modifies: self.data

        Examples:
            If
                    cdata.data = [[1, 2],
                                  [3, 4]]

            then calling cdata.pad_one() will modify the data to

                    cdata.data = [[1, 2, 0],
                                  [3, 4, 0]]

            The dimension of each feature vector is now three.

        This is useful, for example, in any encoding that requires an even number of qubits when the input data has
        an odd number of features.
        """
        # Get an array of zeros to append to self.data
        zeros = np.array([[0]] * self.data.shape[0], dtype=self.data.dtype)

        self.data = np.append(self.data, zeros, axis=1)

    def pad_to_power2(self):
        """Appends zero elements to each data point until the dimension the next highest power of two.

        Modifies: self.data.

        Examples:
            If
                    cdata.data = [[1, 2, 3],
                                  [4, 5, 6]],

            then calling cdata.pad_to_power2() will modify the data to

                    cdata.data = [[1, 2, 3, 0],
                                  [4, 5, 6, 0]].

            The dimension is now a power of two (four).

        This is useful, for example, in a WaveFunctionEncoding, which encodes each feature into an amplitude.
        """
        # Keep appending zeros until the number of features is a power of two
        while self.num_features & (self.num_features - 1) != 0:
            self.pad_one()

    def reset(self):
        """Resets self.data to original input value. Warning: This cannot be undone!

        Modifies: self.data
        """
        self.data = deepcopy(self.raw_data)

    def __getitem__(self, item):
        """Returns the feature vector indexed by item.

        Args:
            item : item
                Index of feature vector to return.
        """
        return self.data[item]


class LabeledCData(CData):
    """Classical data with labels."""
    def __init__(self, data, labels):
        """Initialize classical data with labels.

        Args:
            data [type: numpy array]

        """
        super().__init__(data)
        if callable(labels):
            self.labels = self._compute_labels(labels)
        else:
            # TODO: make sure they're an acceptable type,
            #  then convert them to a standard type
            assert len(labels) == self.num_samples
            self.labels = labels

    def _compute_labels(self, func):
        """Returns an array of labels computed according to
        the input function.

        Args:
            func [type: callable --> bool]
                function that returns a zero or one when called
                on feature vectors
        """
        return np.array([func(x) for x in self.data])

    @property
    def num_classes(self):
        """Returns the number of distinct labels."""
        return len(set(self.labels))

    def train_test_split(self, ratio, shuffle=False):
        """Returns testing and training data."""
        # TODO: take into account the shuffle flag
        assert 0 <= ratio <= 1
        ind = int(ratio * self.num_samples)
        return self.data[:ind], self.data[ind + 1:]

    def keep_data_with_labels(self, labels_to_keep):
        """Modifies data by removing a (data, label) pair if the label is not in labels_to_keep.

        Args:
            labels_to_keep : list<int>
                List of labels to keep (discard all others).

                Examples:
                    labels_to_keep = [0, 1]
                        Discard all (data, label) pairs with labels other than 0 or 1.

                    labels_to_keep = [0, 2]
                        Discard all (data, label) pairs with labels other than 0 or 2.

        REQUIRES:
            Each label in labels_to_keep to be a valid label present in the data set.
        """
        # List to store all indices at which to delete (data, label) pairs
        indices = []

        # Loop over all (data, label) pairs, keeping track of the index
        for (index, label) in enumerate(self.labels):
            # If we don't want to keep this label, add this to the indices to remove
            if label not in labels_to_keep:
                indices.append(index)

        # Remove the (data, label) pairs at these indices
        self.data = np.delete(self.data, indices, axis=0)
        self.labels = np.delete(self.labels, indices)

    def __getitem__(self, item):
        """Override indexing to return data elements."""
        # TODO: question: should this return (data, label) pairs or just data?
        return self.data[item], self.labels[item]


def random_data(num_features, num_samples, labels, seed=None):
    """Returns a CData object with random data."""
    # Seed the random number generator if one is provided
    if seed:
        np.random.seed(seed)

    # Get some random data
    data = np.random.rand(num_samples, num_features)

    # If labels, return a labeled data object
    if labels:
        return LabeledCData(data, labels)

    return CData(data)


def random_data_vertical_boundary(num_samples, seed=None):
    """Returns a CData object with randomly sampled data points
    in the (two-dimensional) unit square. Points left of the line
    x = 0.5 are labeled 0, and points right of the line are labeled 1.

    Args:
        num_samples : int
            Number of data points to return.
    """
    # Seed the random number generator if one is provided
    if seed:
        np.random.seed(seed)

    # Get some random data
    data = np.random.rand(num_samples, 2)

    # Do the labeling
    labels = []
    for point in data:
        if point[0] < 0.5:
            labels.append(0)
        else:
            labels.append(1)

    return LabeledCData(data, labels)


def grid_data(xnum, ynum, labeler=None):
    """Returns a CData object with a two-dimensional (xnum x ynum) grid of points.

    Args:
        xnum : int
            The number of grid points along the x-axis (horizontal axis).

        ynum : int
            The number of grid points along the y-axis (vertical axis).

        labeler : callable
            Function which determines the labels.
    """
    data = []

    for x in np.linspace(0.0, 1.0, xnum):
        for y in np.linspace(0.0, 1.0, ynum):
            data.append([x, y])

    if labeler is None:
        return CData(np.array(data))
    else:
        return LabeledCData(np.array(data), labeler)


def get_iris_setosa_data():
    """Returns a CData object with Iris-Setosa flower data."""
    return LabeledCData(iris.iris_data['data'], iris.iris_data['target'])


def get_mnist_data():
    """Returns a CDdata object with MNIST digits data.

    MNIST data retrieved from training data and labels
    contained in the torchvision datasets module.
    """
    pass
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './data_sets/MNIST/')
    data = datasets.MNIST(
        file_path, train=True, transform=None, target_transform=None, download=False
    )
    # TODO: What are the types here:
    #  Data is required to be a two-dimensional numpy.ndarray of the form
    #  [feature vector #1,
    #   feature vector #2,
    #   ...
    #   feature vector #N]
    #  and features should be a one-dimensional numpy.ndarray of the same length.
    return LabeledCData(data.train_data.numpy(), data.train_labels.numpy())
