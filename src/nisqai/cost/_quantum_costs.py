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

from pyquil.gates import H, CNOT, MEASURE

from nisqai.layer._base_ansatz import BaseAnsatz


class HilbertSchmidtDistance:
    """The quantum circuit for approximating the Hilbert-Schmidt
     distance between two quantum states.
    See QAQC paper.
    """

    def __init__(self, circuit):
        """Initializes a HilbertSchmidtDistance.

        Args:
            circuit : BaseAnsatz
                Circuit that the Hilbert Schmidt test circuit will be applied to.

                Examples:

                    ----[X]----[Y]----[Z]----
                    ----[Y]----[Z]----[X]----
                    ----[X]----[Z]----[Z]----
                    ----[Z]----[Z]----[X]----

                    will produce a Hilbert Schmidt Test circuit of the form

                    ----[H]----@---------[X]----[Y]----[Z]----@---------[H]----[M]
                    ----[H]----|----@----[Y]----[Z]----[X]----|----@----[H]----[M]
                    -----------X----|----[X]----[Z]----[Z]----X----|-----------[M]
                    ----------------X----[Z]----[Z]----[X]---------X-----------[M]

                    Note that the middle part of this circuit is the given circuit above.
        """
        # check inputs
        if not isinstance(circuit, BaseAnsatz):
            raise ValueError("Circuit must be of type BaseAnsatz.")

        # make sure the number of qubits is even
        if circuit.num_qubits % 2 != 0:
            raise ValueError("Invalid number of qubits.")

        # store the input circuit
        self.circuit = circuit

        # get the number of qubits
        self.num_qubits = circuit.num_qubits

        # get the state prep circuit
        self.bell_prep = self._bell_state_prep()

        # get the bell basis measurement circuit
        self.bell_meas = self._bell_basis_measurement()

        # from the entire Hilbert Schmidt Test circuit
        self.hst = self.bell_prep + self.circuit + self.bell_meas

    def _bell_state_prep(self):
        """Returns a generalized Bell state preparation circuit."""
        # get the state prep base ansatz
        prep = BaseAnsatz(self.num_qubits)

        # half the number of qubits, useful for indexing
        half = self.num_qubits // 2

        for q in range(half):
            prep.circuit += [H(q), CNOT(q, q + half)]

        return prep

    def _bell_basis_measurement(self):
        """Returns a Bell basis measurement circuit."""
        # get the state prep base ansatz
        meas = BaseAnsatz(self.num_qubits)

        # half the number of qubits, useful for indexing
        half = self.num_qubits // 2

        for q in range(half):
            meas.circuit += [CNOT(q, q + half), H(q), MEASURE(q)]
            meas.circuit += [MEASURE(q + half)]

        return meas

    def post_process(self):
        """Runs the post processing for the HST."""
        # TODO: implement


class DipTest(BaseAnsatz):
    """Diagonal inner product circuit ansatz."""
    # TODO: complete class!
