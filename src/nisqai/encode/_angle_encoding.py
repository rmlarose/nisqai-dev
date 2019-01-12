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

from nisqai.layer._base_ansatz import BaseAnsatz
from nisqai.data._cdata import CData, LabeledCData


class AngleEncoding():
    """AngleEncoding class."""

    def __init__(self, data, encoder, feature_map):
        """Inititiate an AngleEncoding class."""
        # TODO: better error checking
        assert isinstance(data, (CData, LabeledCData))