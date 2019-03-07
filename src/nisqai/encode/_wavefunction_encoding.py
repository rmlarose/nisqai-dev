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

from numpy import array, cos, sin, exp, dot, identity, isclose
from numpy import identity, delete, linalg, matmul, random, log2, ceil, ndarray # functions yousif used
from pyquil import Program

from pyquil.quil import DefGate


def WavefunctionEncoding(x):
    """WavefunctioneEncoding class. Encode a vector |x> directly via

    |psi> = |x> / ltwonorm(|x>).


    With this encoding, an n length vector needs n qubits.

    Args:
        x : list
            List of coefficients for the state in canonical binary ordering 00..0, 00..01, 00..10, 00..11, ...

        num_qubits : int
            The number of qubits in the circuit.

        data : ...
    """
    # Make sure x is a list or array.
    assert isinstance(x, list) or isinstance(x, ndarray), "Input vector must be a list or numpy array."

    # If length of vector is not already a power of 2, make it so.
    if ( ( len(x) & ( len(x) - 1 ) ) == 1 ) or len(x) == 0 :
        new_len = 2**( ceil( log2( len(x) ) ) )
        len_diff = new_len - len(x)
        # In case x was input as an array.
        x = list(x) + [0]*len_diff
    
    
    # Normalize vector.
    x = array(x) / linalg.norm(x)

    # We'll use Gram-Schmidt to compute 2^{|x|} - 1 orthogonal rows to x, 
    # starting with x and standard basis vectors.
    # If rows a square matrix form an orthonormal basis, 
    # columns will too. Thus we'll have a unitary matrix.

    standard_basis = identity(len(x))

    # Delete a row so you only have 2^{|x|} - 1 orthogonal vectors
    # But first, make sure vector input is not in span of one of 
    # the vectors used for Gram-Schmidt. (Then starting set won't be a basis.)
    #
    # TODO: This method is janky and should be replaced.
    flag = 0
    for i in range(len(x)):
        if abs(1 - abs(dot(x.conj(), standard_basis[i]))) < 1e-2:
            standard_basis = delete(standard_basis, i, 0)
            flag = 1
            break
    if flag == 0:
        standard_basis = delete(standard_basis, 0, 0)

    U = []
    a = x
    for k in range(len(x)):
        for i in range(k):
            a += dot(array(U[i]).conj(),standard_basis[k-1])*array(U[i])
        if k != 0:
            a = standard_basis[k-1] - a
        a = a / linalg.norm(a)
        U.append(a)
        a = 0
    return array(U).T

def make_program(U):
    num_qubits    = int(log2(len(U)))
    gate_def      = DefGate("U", U)
    U_gate        = gate_def.get_constructor()
    qubit_indices = range(num_qubits)
    p             = Program(gate_def, U_gate(*tuple(qubit_indices)))
    return p
