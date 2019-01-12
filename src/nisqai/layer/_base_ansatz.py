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

"""Module with basic ansatz class definitions to be inherited by other
ansatz classes.
"""

from pyquil import Program, get_qc, list_quantum_computers

REAL_MEM_TYPE = "REAL"
BIT_MEM_TYPE = "BIT"

class BaseAnsatz():
    """Base ansatz for all ansatz classes."""
    
    def __init__(self, num_qubits):
        self._num_qubits = num_qubits
        self.circuit = Program()

    @property
    def num_qubits(self):
        """Returns the number of qubits in the ansatz."""
        return self._num_qubits

    def depth(self, quantum_computer):
        """Computes the depth of the circuit ansatz.
        
        Here, the depth is the maximum number of "incompressible" operations
        over all qubits.
        EDIT: Since it's not obvious how to efficiently implement the above,
        let depth be the number of gates after compilation.
        """
        # TODO: make gate_alphabet an argument (probably write a gate_alphabet
        # class)
        qc_list = list_quantum_computers()
        assert ( quantum_computer.startswith(tuple(qc_list)) or 
            ( quantum_computer[0:-5].isdigit() and quantum_computer[-5::] == 'q-qvm') )
        qc = get_qc(quantum_computer)
        p = self.circuit
        np = qc.compiler.quil_to_native_quil(p)
        return len(np.instructions)
        

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

    def clear_circuit(self):
        """Clears all instructions in the circuit ansatz."""
        self.circuit = Program()

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
        