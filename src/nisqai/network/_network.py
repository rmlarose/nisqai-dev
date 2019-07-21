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

from nisqai.measure import MeasurementOutcome

from numpy import ndarray

from pyquil import get_qc
from pyquil.api import QuantumComputer

# TODO: This should be updated to something like
#  from nisqai.trainer import this_optimization_method
#  for now this is just for simplicity
# from scipy.optimize import minimize
from nisqai.optimize import minimize


class Network:
    """Network class."""

    def __init__(self, layers, computer, predictor=None):
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

            predictor : Callable
                Function that inputs a bit string and outputs a label
                (i.e., either 0 or 1) representing the class.
        """
        # TODO: check if ordering of layers is valid

        # Store the layers and individual elements
        # TODO: are both needed? which is better?
        self._layers = layers
        self._encoder = layers[0]
        self._ansatz = layers[1]
        self._measurement = layers[2]

        # Store the computer backend
        if type(computer) == str:
            self.computer = get_qc(computer)
        elif type(computer) == QuantumComputer:
            self.computer = computer
        else:
            raise TypeError

        # Number of data points
        self.num_data_points = self._encoder.data.num_samples

        # TODO: Make sure the predictor function is valid (returns 0 or 1)
        self.predictor = predictor

    def _build(self, data_ind):
        """Builds the network as a sequence of quantum circuits."""
        # TODO: what about multicircuit networks?
        #  note 2/4/19: I think this could be handled with another class

        # Grab the initial encoder circuit for the given index
        circuit = self._encoder[data_ind]

        # Add all other layers
        # TODO: allow self.layers to take sublists
        #  for example, [encoder, [layer1, layer2, layer3, ...], measure]
        #  this could make it easier to build networks using, say, list comprehensions
        for ii in range(1, len(self._layers)):
            circuit += self._layers[ii]

        # Order the given circuit and return it
        circuit.order()
        return circuit

    def compile(self, index, shots):
        """Returns the compiled program for the data point
        indicated by the index.

        Args:
            index : int
                Index of data point.

            shots : int
                Number of times to run the circuit.
        """
        # Get the right program to compile. Note type(program) == BaseAnsatz.
        program = self._build(index)

        # Compile the program to the appropriate computer
        return program.compile(self.computer, shots)

    def propagate(self, index, angles=None, shots=1000):
        """Runs the network (propagates a data point) and returns the circuit result.

        Args:
            index : int
                Specifies the index of the data point to propagate.

            angles : Union[dict, list]
                Angles for the unitary ansatz.

            shots : int
                Number of times to execute the circuit.
        """
        # Get the compiled executable instructions
        executable = self.compile(index, shots)

        # Use the memory map from the ansatz parameters
        if angles is None:
            mem_map = self._ansatz.params.memory_map()
        else:
            mem_map = self._ansatz.params.update_values_memory_map(angles)

        # Run the program and store the raw results
        output = self.computer.run(executable, memory_map=mem_map)

        # Return a MeasurementOutcome of the results
        return MeasurementOutcome(output)

    def predict(self, index, angles=None, shots=1000):
        """Returns the prediction of the data point corresponding to the index.

        Args:
            index : int
                Specifies the index of the data point to get a prediction of.

            angles : Union[dict, list]
                Angles for the unitary ansatz.

            shots : int
                Number of times to execute the circuit.
        """
        # Propagate the network to get the outcome
        output = self.propagate(index, angles, shots)

        # Use the predictor function to get the prediction from the output
        # TODO: NOTE: This is not compatible with classical costs such as cross entropy.
        prediction = self.predictor(output)

        # Return the prediction
        return prediction

    def cost_of_point(self, index, angles=None, shots=1000):
        """Returns the cost of a particular data point.

        Args:
            index : int
                Specifies the data point.

            angles : Union(dict, list)
                Angles for the unitary ansatz.

            shots : int
                Number of times to execute the circuit.
        """
        # Get the network's prediction of the data point
        prediction = self.predict(index, angles, shots)

        # Get the actual label of the data point
        label = self._encoder.data.labels[index]

        # TODO: Generalize to arbitrary cost functions.
        #  Input a cost function into the Network, then use this.
        return int(prediction != label)

    def cost(self, angles, shots=1000):
        """Returns the total cost of the network at the given angles.

        Args:
            angles : Union(dict, list)
                Angles for the unitary ansatz.

            shots : int
                Number of times to execute the circuit.

        Returns : float
            Total cost of the network.
        """
        # Variable to store the cost
        val = 0.0

        # Add the cost for each data point
        for ii in range(self.num_data_points):
            val += self.cost_of_point(ii, angles, shots)

        # Return the total normalized cost
        return val / self.num_data_points

    def train(self, initial_angles, trainer="COBYLA", shots=1000, **kwargs):
        """Adjusts the parameters in the Network to minimize the cost.

        Args:
            initial_angles : Union[dict, list]

            trainer : callable
                Optimization function used to minimize the cost.
                Defaults to "COBYLA"

            shots: int, number of times to run the circuit(s).
                defaults to 1000.

        kwargs: 
            Keyword arguments sent into the `options` argument in the
            nisqai.optimize.minimize method. For example:
                >>> Network.train(initial_angles, trainer="Powell", maxfev=100)
            will call
                >>> nisqai.optimize.minimize(cost, initial_angles, 
                                             method="Powell", options=dict(maxfev=100))
            This is consistent with how scipy.optimize.minimize is formatted.
        """
        # Define the objective function
        obj = lambda angles: self.cost(angles=angles, shots=shots)

        # Call the trainer
        res = minimize(obj, initial_angles, method=trainer, options=kwargs)

        # TODO: Define a NISQAI standard output for trainer results
        return res

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
