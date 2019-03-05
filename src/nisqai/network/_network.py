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

from pyquil import get_qc
from pyquil.api import QuantumComputer


class Network:
    """Network class."""

    def __init__(self, layers, computer):
        """Initializes a network with the input layers.

        Args:
            layers : iterable
                Iterable object of network elements.

                Examples:
                    layers = [DenseAngleEncoding, ProductAnsatz, Measurement]
                    leads to a network of the form
                        ----[State Prep]----[Gate]----[Measure]----

                Network elements must be in a valid ordering to create a network.

                Criteria:
                    (1) Must start with an encoding ansatz.
                    (2) Must end with a measurement ansatz.
                    (3) Any number of unitary ansatze can be implemented in between.
                    (4) If network continues after measurement, an encoding ansatz
                        must follow a measurement ansatz.

            computer : Union[str, pyquil.api.QuantumComputer]
                Specifies which computer to run the network on.

                Examples:
                    "Aspen-1-2Q-B"
                    "1q-qvm"
                    "5q-qvm"
        """
        # TODO: check if ordering of layers is valid

        # store the layers and individual elements
        # TODO: are both needed? which is better?
        self._layers = layers
        self._encoder = layers[0]
        self._ansatz = layers[1]
        self._measurement = layers[2]

        # store the computer backend
        if type(computer) == str:
            self.computer = get_qc(computer)
        elif type(computer) == QuantumComputer:
            self.computer = computer
        else:
            raise TypeError

    def _build(self, data_ind):
        """Builds the network as a sequence of quantum circuits."""
        # TODO: what about multicircuit networks?
        # note 2/4/19: I think this could be handled with another class

        # grab the initial encoder circuit for the given index
        circuit = self._encoder[data_ind]

        # add all other layers
        # TODO: allow self.layers to take sublists
        # TODO: for example, [encoder, [layer1, layer2, layer3, ...], measure]
        # TODO: this could make it easier to build networks using, say, list comprehensions
        for ii in range(1, len(self._layers)):
            circuit += self._layers[ii]

        # order the given circuit and return it
        circuit.order()
        return circuit

    def propagate(self, index, shots):
        """Runs the network and returns the result, using the current parameters.

        Args:
            index : int
                Specifies the index of the data point to propagate.

            shots : int
                Number of times to execute the circuit.
        """
        # get the compiled executable instructions
        executable = self.compile(index, shots)

        # use the memory map from the ansatz parameters
        mem_map = self._ansatz.params.memory_map()

        # run the program
        return self.computer.run(executable, memory_map=mem_map)

    def compile(self, index, shots):
        """Returns the compiled program for the data point
        indicated by the index.

        Args:
            index : int
                Index of data point.

            shots : int
                Number of times to run the circuit.
        """
        # get the right program to compile
        program = self._build(index)

        # compile the program to the appropriate computer
        return program.compile(self.computer, shots)

    def train(self, cost, shots):
        """Adjusts the parameters in the Network to minimize the cost.

        Args:
            cost : Callable
                Function that returns the cost of the Network.

            shots : int
                Number of times to run the circuit(s).
        """
        print("in Network.train, len encoder =", len(self._encoder))
        #executable = self.compile()

    def __getitem__(self, index):
        """Returns the network with state preparation for the data
        point indicated by item.

        Args:
            index : int
                Index of data point.
        """
        return self._build(index)

    def __str__(self):
        """Returns the circuit for the zeroth data point."""
        # TODO: return a text drawing of the network
        return self[0]
