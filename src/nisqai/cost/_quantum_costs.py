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
from nisqai.measure._measure import Measurement
import numpy as np
from pyquil.api import get_qc, local_qvm


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


class Observable(BaseAnsatz):
    """Observable class."""

    def __init__(self, num_qubits, qubits_to_measure):
        """Initializes a circuit to compute expectation value Hermitian observable of all qubits.

        Args:
            qubits_to_measure : iterable
                Indices of qubits to perform measurements on.

            basis_gate : string
                Basis to measure qubits in.

            basis_angle : float
                Rotation angle for parameterised single qubit gates.
        """
        if num_qubits <= 0:
            raise ValueError("Invalid number of qubits.")

        super().__init__(num_qubits)
        self.measured_qubits = list(qubits_to_measure)

    def add_single_qubit_observable_meas(self, observable=None):
        """Add measurement instructions to measure a single qubit in particular basis 
            for desired observable
         Args:
            qubits_to_measure : iterable
                Iterable of qubits for which to measure observable.

            observable : string or list
                Observable to measure of the qubit, qubit_index. 
                Should be of same form as Measurement change_basis method argument.

        """
        # Default measure Z observable
        meas_circ  = Measurement(self.num_qubits, self.measured_qubits)
        if observable is not None:
            if type(observable) is not str:
                raise TypeError("Invalid observable format")
            else:
                if observable.lower() == 'z':
                    pass

                elif observable.lower() == 'x':
                    meas_circ.change_basis("H")

                elif observable.lower() == 'y':
                    meas_circ.change_basis(['XY', np.pi/2])

                else:
                    meas_circ.change_basis(observable)

        self.circuit += meas_circ.circuit

    def compute_observables_exp(self, computer, num_shots, observable=None):
        """Returns Expectation value of observable w.r.t. all qubits 
        Args:
            computer : string
                String indicating Quantum Computer to run on.

            num_shots: int
                Number of measurement shots to take when computing observables.

            observable : string or list
                Observable to measure of the qubit, qubit_index. 
                Should be of same form as Measurement change_basis method argument.

        Returns:
            expectation_value : np.ndarray
            Expectation values of each qubit measured with respect to specified
            observable.
        """
        if type(num_shots) is not int:
            raise TypeError("Number of measurement shots must be an integer")
        if num_shots < 0:
            raise ValueError("Number of measurement shots can't be negative")
         
        self.num_meas_shots = num_shots
        self.add_single_qubit_observable_meas(observable)
        qc = get_qc(computer)

        compiled_prog = self.compile(computer, self.num_meas_shots)
      
        meas_results = qc.run(compiled_prog)

        expectation_value = np.zeros(len(self.measured_qubits))
        for shot in range(len(meas_results)):
            for qubit in self.measured_qubits:
                if meas_results[shot][qubit] == 0:
                    expectation_value[qubit] += 1/num_shots
                else:
                    expectation_value[qubit] += -1/num_shots

        return expectation_value

    def sum_observables(self, computer, num_shots, observable=None):
        """Returns sum of expectation value of expectation values of 
        observables with respect to all qubits.

        Args:
            computer : string
                String indicating Quantum Computer to run on.

            num_shots: int
                Number of measurement shots to take when computing observables.

            observable : string or list
                Observable to measure of the qubit, qubit_index. 
                Should be of same form as Measurement change_basis method argument.

        Returns: float
            Sum of expectation values of product observable on given qubits.
        """
        expectation_values =  self.compute_observables_exp(computer, num_shots, observable)
        
        return np.sum(expectation_values)

        #TODO: Add general costs related to observables; Expectation of general Hamiltonians etc.

