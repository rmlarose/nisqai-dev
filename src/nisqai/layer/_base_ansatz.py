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

"""Basic ansatz class to be inherited by other ansatz classes."""

from pyquil import Program, get_qc, list_quantum_computers
from pyquil.quil import percolate_declares
from pyquil.quilbase import Gate

REAL_MEM_TYPE = "REAL"
BIT_MEM_TYPE = "BIT"


class BaseAnsatz:
    """Base ansatz for all ansatz classes."""
    
    def __init__(self, num_qubits):
        """Initialize a BaseAnsatz."""
        self._num_qubits = num_qubits
        self.circuit = Program()

    @property
    def num_qubits(self):
        """Returns the number of qubits in the ansatz."""
        return self._num_qubits

    def compile(self, computer, shots=1000):
        """Returns a compiled circuit for a given quantum computer.

        Args:
            computer : str
                Quantum computer to compile to.

            shots : int
                Number of times to run the circuit.
        """
        # make sure the quantum computer is valid
        qc_list = list_quantum_computers()
        if not (computer.startswith(tuple(qc_list)) or
                (computer[0:-5].isdigit() and computer[-5::] == "q-qvm")):
            raise ValueError("Invalid computer type.")

        # compile to the given computer
        computer = get_qc(computer)
        self.circuit.wrap_in_numshots_loop(shots=shots)
        return computer.compiler.quil_to_native_quil(self.circuit)

    def depth(self, computer):
        """Returns the depth of the circuit ansatz given by
        the total number of gates after compilation to a computer.

        Args:
            computer : str
                Valid string specifying a quantum computer to compile to.

        Returns:
            Number of gates after compilation to the computer.
        """
        return len([obj for obj in self.compile(computer, 1) if type(obj) == Gate])

    def num_ops(self, qubits):
        """Returns the total number of operations over a subset of qubits
        in the circuit ansatz.
        """
        # TODO: complete method
        pass

    def add_gates(self, qubit, gates):
        """Adds a list of gates that act on a qubit.
        
        Args:
            qubit [type: int]
                Index of the qubit to be rotated.
            
            gates [type: list]
                Single qubit gates to act on the qubit.
        
        Modifies:
            self.circuit
        """
        self.circuit.inst(
            [gate(qubit) for gate in gates]
            )

    def add_layer(self, gate):
        """Adds the gate to every qubit in the circuit."""
        for q in range(self._num_qubits):
            self.circuit.inst(gate(q))

    def add_at(self, gate, qubit_indices):
        """Adds the gate to self.circuit at each index in qubit_indices.

        Args:
            gate : pyquil gate operation
                Gate to add to the circuit.

            qubit_indices : iterable
                Indices of which qubit to add the gate at.
        """
        gates = [gate(ind) for ind in qubit_indices]
        self.circuit.inst(gates)

    def clear_circuit(self):
        """Clears all instructions in the circuit ansatz."""
        # TODO: rewrite clearing only instructions, not DEFGATEs or PARAMS
        self.circuit = Program()

    def order(self):
        """Orders Quil instructions into a nominal form."""
        # TODO: define nominal form and add more ordering conditions
        # TODO: right now, this just means all DECLARE statements are at the top
        percolate_declares(self.circuit)

    def __str__(self):
        """Returns a circuit diagram."""
        # TODO: complete method
        # probably want to write a TextDiagramDrawer class like in Cirq
        return self.circuit.__str__()

    def __add__(self, ansatz):
        # TODO: make sure this works with all derived ansatz types
        # e.g., make sure ProductAnsatz__sum__ returns a ProductAnsatz, etc.
        assert isinstance(ansatz, BaseAnsatz)
        assert self._num_qubits == ansatz.num_qubits
        new = BaseAnsatz(self._num_qubits)
        new.circuit = self.circuit + ansatz.circuit
        return new
