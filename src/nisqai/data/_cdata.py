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

from numpy import amax


class CData():
    """Classical data class."""

    def __init__(self, data):
        """Initialize a BaseCData object.
        
        Args:
            data [type: numpy array]
                data values, shape should be (samples, features).
        """
        self.data = data
        self.num_samples, self.num_features = data.shape

    def normalize(self):
        """Divides all data by the max element.

        Modifies self.data in place.
        """
        self.data /= amax(self.data)

    def reduce_features(self, features_to_keep):
        """Performs (classical) principal component analysis
         on the data and keeps the desired number of features.

        Args:
            features_to_keep [type: int or float]
                if int, keeps exactly this many features.
                if float, keeps this ratio of features.

        Examples:
            reduce_features(5) --> keeps top 5 features after PCA
            reduce_features(0.2) --> keeps top 20% of features after PCA
        """
        # TODO: complete method
        pass


class LabeledCData(CData):
    """Classical data with labels."""
    def __init__(self, data, labels):
        super().__init__(data)
        self.labels = labels


