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

from nisqai.encode._angle_encoding import AngleEncoding
from nisqai.encode._dense_angle_encoding import DenseAngleEncoding
from nisqai.encode._binary_encoding import BinaryEncoding
from nisqai.encode._plus_minus_encoding import PlusMinusEncoding
from nisqai.encode._wavefunction_encoding import WavefunctionEncoding

from nisqai.encode._feature_maps import direct, nearest_neighbor
from nisqai.encode._encoders import angle_simple_linear, linear_encoder
