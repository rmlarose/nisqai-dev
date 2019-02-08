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

from nisqai.utils._program_utils import order


class Network:
    """Network class."""

    def __init__(self, layers, computer):
        """Initializes a network with the input layers.

        Args:
            layers : iterable
                Iterable object of network elements.

                Examples:
                    layers = [product ansatz on one qubit, measurement]
                    leads to a network of the form
                        --[Gate]--[Measure]--

                Network elements must be in a valid ordering to create a network.

                Criteria:
                    (1) Must start with an encoding ansatz.
                    (2) Must end with a measurement ansatz.
                    (3) Any number of unitary ansatze can be implemented in between.
                    (4) If network continues after measurement, an encoding ansatz
                        must follow a measurement ansatz.

            computer : string
                Specifies which computer to run the network on.

                Examples:
                    "Aspen-1-2Q-B"
        """
        # TODO: check if ordering of layers is valid

        # TODO: check if computer is valid

        self.layers = layers
        self.computer = computer

    def _build(self, data_ind):
        """Builds the network as a sequence of quantum circuits."""
        # TODO: what about multicircuit networks?
        # note 2/4/19: I think this could be handled with another class

        # grab the initial encoder circuit for the given index
        network = self.layers[0][data_ind]

        # add all other layers
        # TODO: allow self.layers to take sublists
        # TODO: for example, [encoder, [layer1, layer2, layer3, ...], measure]
        # TODO: this could make it easier to build networks using, say, list comprehensions
        for ii in range(1, len(self.layers)):
            network += self.layers[ii]

        network.order()
        return network

    def propagate(self, parameters, trials):
        """Runs the network and returns the result.

        Args:
            parameters : list
                Any parameters needed for the network.

            trials : int
                Number of times to run each circuit.
        """
        pass

    def compile(self, index):
        """Returns the compiled program for the data point
        indicated by the index.

        Args:
            index : int
                Index of data point.
        """
        # get the right program to compile
        program = self._build(index)

        # compile the program to the appropriate computer
        return program.compile(self.computer)

    def __getitem__(self, index):
        """Returns the network with state preparation for the data
        point indicated by item.

        Args:
            index : int
                Index of data point.
        """
        return self._build(index)

