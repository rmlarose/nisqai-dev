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
from numpy import identity, delete, linalg, matmul, random, log2 # functions yousif used
from pyquil import Program

from pyquil.quil import DefGate


def WavefunctionEncoding(x):
    """WavefunctioneEncoding class. Encode a vector |x> directly via

    |\psi> = |x> / \ltwonorm(|x>).


    With this encoding, an n length vector needs n qubits.

    Args:
        x : list
            List of coefficients for the state in canonical binary ordering 00..0, 00..01, 00..10, 00..11, ...

        num_qubits : int
            The number of qubits in the circuit.

        data : ...
    """
    # Use Gram-Schmidt to compute 2^{|x|} - 1 orthogonal rows to x, starting with x and standard basis vectors.
    # If rows a square matrix form an orthonormal basis, columns will too. Thus we'll have a unitary matrix.

    # Normalize vector
    x = array(x) / linalg.norm(x)

    standard_basis = identity(len(x))

    # Delete a row so you only have 2^{|x|} - 1 orthogonal vectors
    # But first, make sure vector input is not in span of one of the vectors used for Gram-Schmidt. (Then starting set won't be a basis.)
    flag = 0
    for i in range(len(x)):
        if abs(1 - abs(dot(x.conj(), standard_basis[i]))) < 1e-2:
            standard_basis = delete(standard_basis, i, 0)
            #print(standard_basis)
            flag = 1
            break
    if flag == 0:
        standard_basis = delete(standard_basis, 0, 0)
    
    #print(standard_basis)

    U = []
    a = x
    for k in range(len(x)):
        for i in range(k):
            #print("here",array(U[i]).conj())
            a += dot(array(U[i]).conj(),standard_basis[k-1])*array(U[i])
        if k != 0:
            a = standard_basis[k-1] - a
        a = a / linalg.norm(a)
        U.append(a)
        a = 0
    return array(U).T

def make_program(U):
    U_definition = DefGate("U_x", U)
    U_gate = U_definition.get_constructor()
    num_qubits = int(log2(len(U)))
    # p = Program(U_definition, U_gate( QUBIT TUPLE GOES HERE ))
    # return p
