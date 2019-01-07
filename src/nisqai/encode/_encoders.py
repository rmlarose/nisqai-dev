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

from math import pi


def linear_encoder(coeffs, feature_vector):
    """Returns a linear combination of all features."""
    assert len(coeffs) == len(feature_vector)
    return sum(coeffs[i] * feature_vector[i] for i in range(len(feature_vector)))


def angle_simple_linear(feature_vector):
    """Returns the "simple linear encoding" of the feature vectors

    theta = pi * feature_vector[0]
    phi = 2 * pi * feature_vector[1]
    """
    return (pi * feature_vector[0], 2 * pi * feature_vector[0])
