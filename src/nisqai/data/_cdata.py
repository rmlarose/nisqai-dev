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

from numpy import array, random, float64
from numpy.linalg import norm as LAnorm
from copy import deepcopy
import torch
import os
import torchvision
from nisqai.data.data_sets import iris


class CData:
    """Classical data class."""

    def __init__(self, data):
        """Initialize a BaseCData object.

        Args:
            data [type: numpy array]
                data values, shape should be (samples, features).
        """
        # TODO: allow data to be a pandas dataframe, 2d list, and
        # other possible data types people would just want to throw
        # into the class without thinking about it
        # for pandas dataframes, just need to convert it to an array
        self.raw_data = data
        self.data = deepcopy(self.raw_data)
        shape = data.shape
        self.num_samples = shape[0]
        # to get total number of features from tensor data
        total = 1
        for x in shape[1::]:
            total *= x
        self.num_features = total

    def scale_features(self, method):
        """ Performs feature scaling on data.

        Args:
            method [type: string]
                specifies desired feature scaling method
                * 'min-max norm'
                x' = (x - min(x))/(max(x) - min(x))
                * 'mean norm'
                x' = (x-avg(x))/(max(x) - min(x))
                * 'standardize'
                x' = (x - avg(x))/sigma
                * 'L2 norm'
                x' = x / L2norm(x)
                * 'L1 norm'
                x' = x / L1norm(x)
        """
        data = self.data
        if method == 'min-max norm':
            mmin = data.min(axis=0)
            mmax = data.max(axis=0)
            self.data = (data - mmin) / (mmax - mmin)
        elif method == 'mean norm':
            mean = data.mean(axis=0)
            mmin = data.min(axis=0)
            mmax = data.max(axis=0)
            self.data = (data - mean) / (mmax - mmin)
        elif method == 'standardize':
            mean = data.mean(axis=0)
            # ddof = 1 gives sample sd
            sd = data.std(axis=0, ddof=1)
            self.data = (data - mean) / sd
        elif method == 'L2 norm':
            norm = LAnorm(data, axis=0)
            self.data = data / norm
        elif method == 'L1 norm':
            L1norm = sum(abs(data))
            self.data = data / L1norm

    def reduce_features(self, kfeat):
        """Performs (classical) principal component analysis
         on the data and keeps the desired number of features.

        Args:
            kfeat [type: float]
                keeps this ratio of features.

        Examples:
            reduce_features(0.2) --> keeps top 20% of features after PCA
        """
        # cast data as PyTorch tensor
        data = torch.from_numpy(self.data)

        # preprocess the data
        data_mean = torch.mean(data, 0)
        data = data - data_mean.expand_as(data)

        # do svd
        U, S, V = torch.svd(torch.t(data))
        return torch.mm(data, U[:, :kfeat])

    def __getitem__(self, item):
        """Override indexing to return data elements."""
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
            # then convert them to a standard type
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
        return array([func(x) for x in self.data])

    def train_test_split(self, ratio, shuffle=False):
        """Returns testing and training data."""
        # TODO: take into account the shuffle flag
        assert ratio >= 0 and ratio <= 1
        ind = int(ratio * self.num_samples)
        return self.data[:ind], self.data[ind + 1:]

    def __getitem__(self, item):
        """Override indexing to return data elements."""
        # TODO: question: should this return (data, label) pairs or just data?
        return self.data[item], self.labels[item]


def random_data(num_features, num_samples, labels, dtype=float64, seed=None):
    """Returns a CData object with random data."""
    # seed the random number generator if one is provided
    if seed:
        random.seed(seed)

    # get some random data
    data = random.rand(num_samples, num_features)

    # if labels, return a labeled data object
    if labels:
        return LabeledCData(data, labels)

    return CData(data)


def get_iris_setosa_data():
    """Returns a CData object with Iris-Setosa flower data."""
    return LabeledCData(iris.iris_data['data'], iris.iris_data['target'])


def get_mnist_data():
    """Returns a CDdata object with MNIST digits data.

    MNIST data retrieved from training data and labels
    contained in the torchvision datasets module.
    """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './data_sets/MNIST/')
    data = torchvision.datasets.MNIST(file_path, train=True,
                                      transform=None, target_transform=None, download=False)
    return LabeledCData(data.train_data.numpy(), data.train_labels.numpy())
