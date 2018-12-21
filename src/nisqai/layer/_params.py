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

class Param():
    """Class for working with parameters in a circuit."""

    def __init__(self, num_qubits, depth):
        self._num_qubits = num_qubits
        self._depth = depth
    
    """TODO: efficient data structure for storing and working with parameters.
    
    Should have:
        
        (1) A linear array for optimization methods.
        
        (2) A shape that makes it easy to index for adding parameters
            into circuits.
            
            For example, the circuit
            
            ----[Ry(theta)]----[Rz(phi)]----
            ----[Rz(lambda)]---[Ry(eta)]----
            
            could have parameters stored in a two-dimensional array:
                
                [[theta, phi],
                 [lambda, eta]]
            
            to make it simple to add them to the circuit.
    """