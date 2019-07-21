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


class WaveFunctionEncoding:
    """WaveFunctionEncoding class. Encode a vector |x> directly via |psi> = |x> / |<x|x>|^2.

    Warning: Not NISQ!
    """
    def __init__(self, cdata, auto_pad=False):
        """Initialize a WaveFunctionEncoding.

        Args:
            cdata : Union[CData, LabeledCData]
                Classical data to encode in a wavefunction. For each data vector |x>,

            auto_pad : bool
                If True, appends zero elements to the data feature vectors until the dimension is a power of two.
                If False and the dimension is not a power of two, an error will be thrown.
                If False and the dimension is a power of two, no error is thrown.
        """
        # Type checking
        assert isinstance(cdata, (CData, LabeledCData))

        # Make sure the number of features is a power of two
        if auto_pad:
            cdata.pad_to_power2()
        else:
            if cdata.num_features & (cdata.num_features - 1) != 0:
                raise ValueError(
                    "The number of features in the data is not a power of two." +
                    "To fix this, set auto_pad=True OR try cdata.pad_to_power2()."
                )

        # Store the data
        self.data = cdata

        # Determine the number of qubits from the input data
        self.num_qubits = self._compute_num_qubits()

        # List to hold the circuit for each data point
        self.circuits = [BaseAnsatz(self.num_qubits) for _ in range(self.data.num_samples)]

        # Write each circuit
        for ind in range(len(self.circuits)):
            self._write_circuit(ind)

    def _compute_num_qubits(self):
        """Computes the number of qubits needed for the encoding."""
        return int(log2(self.data.num_features))

    def _write_circuit(self, feature_vector_index):
        """Writes the circuit for the given feature vector index."""
        # Build a unitary for the feature vector
        unitary = self._make_unitary(feature_vector_index)

        # Compile this unitary to a program
        program = self._compile_unitary(unitary)

        # Write this program into the circuit of the ansatz
        self.circuits[feature_vector_index].circuit = program

    def _compile_unitary(self, unitary):
        """Compiles the unitary to a program.

        Args:
            unitary : numpy.ndarray
                Unitary matrix to compile.

        Returns:
            A pyquil.Program with gates that build up the unitary.
        """
        # Define the gate from the unitary
        gate = DefGate("U", unitary)

        # Get the constructor
        U_gate = gate.get_constructor()

        # Get the qubit indices
        qubit_indices = range(self.num_qubits)

        # Return the program
        return Program(gate, U_gate(*tuple(qubit_indices)))

    def _make_unitary(self, feature_vector_index):
        """Returns a unitary matrix built by starting with a feature vector and using Gram Schmidt orthogonalization.

        Args:
            feature_vector_index : int
                Index of feature vector in self.data.

        Return type; numpy.ndarray
        """
        x = self.data.data[feature_vector_index]

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
                a += dot(array(U[i]).conj(), standard_basis[k - 1]) * array(U[i])
            if k != 0:
                a = standard_basis[k - 1] - a
            a = a / linalg.norm(a)
            U.append(a)
            a = 0

        return array(U).T

    def __getitem__(self, ind):
        """Returns the circuit for the data point indexed by ind."""
        if not isinstance(ind, int):
            raise TypeError("Argument ind must be of type int.")
        return self.circuits[ind]
