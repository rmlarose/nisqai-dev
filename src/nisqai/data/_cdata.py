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

from numpy import amax, array, average, random, float64
from copy import deepcopy

class CData():
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
        self.num_samples, self.num_features = data.shape
        self._process()

    def _normalize(self):
        """Divides all data by the max element."""
        # TODO: complete method
        #self.data /= amax(self.data)
        pass

    def _center(self):
        """Subtracts the mean of the data from each entry."""
        # TODO: complete method
        #self.data -= average(self.data)
        pass

    def _process(self):
        """Processes data."""
        self._center()
        self._normalize()

    def reduce_features(self, features_to_keep):
        """Performs (classical) principal component analysis
         on the data and keeps the desired number of features.

        Args:
            features_to_keep [type: float]
                keeps this ratio of features.

        Examples:
            reduce_features(0.2) --> keeps top 20% of features after PCA
        """
        # TODO: complete method
        pass


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