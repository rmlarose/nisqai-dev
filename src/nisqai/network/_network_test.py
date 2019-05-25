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

from numpy import array

import unittest

from pyquil import Program, get_qc
from pyquil.api import QuantumComputer

from nisqai.layer._base_ansatz import BaseAnsatz
from nisqai.network._network import Network
from nisqai.data._cdata import random_data, CData, LabeledCData
from nisqai.encode._dense_angle_encoding import DenseAngleEncoding
from nisqai.encode._binary_encoding import BinaryEncoding
from nisqai.layer._product_ansatz import ProductAnsatz
from nisqai.measure._measure import Measurement
from nisqai.encode._encoders import angle_simple_linear
from nisqai.encode._feature_maps import nearest_neighbor


class TestNetwork(unittest.TestCase):
    """Unit tests for Network class."""

    def test_simple(self):
        """Tests if a Network can be instantiated."""
        # Get components for a simple network
        cdata = random_data(num_features=2, num_samples=100, labels=[1 if x < 50 else 0 for x in range(100)])
        encoder = DenseAngleEncoding(cdata, angle_simple_linear, nearest_neighbor(2, 1))
        layer = ProductAnsatz(1)
        measure = Measurement(1, range(1))
        computer = "1q-qvm"

        # make the network
        qnn = Network([encoder, layer, measure], computer)

        # check some basics
        self.assertEqual(type(qnn.computer), QuantumComputer)

    def test_build_basic(self):
        """Tests building a simple Network."""
        # Get the components for a network
        data = array([[0, 1], [1, 0]])
        cdata = CData(data)
        encoder = BinaryEncoding(cdata)
        layer = ProductAnsatz(2)
        measure = Measurement(2, [0])

        # Make the network
        qnn = Network([encoder, layer, measure], computer="2q-qvm")

        # Build each circuit for the network
        net0 = qnn._build(0)
        net1 = qnn._build(1)

        # Check that each circuit is a BaseAnsatz
        self.assertEqual(type(net0), BaseAnsatz)
        self.assertEqual(type(net1), BaseAnsatz)

    def test_build_multiple_ansatze(self):
        """Tests building a Network with multiple sequential unitary ansatze."""
        # TODO: options for adding ansatze together -- keep parameters the same or define new ones?
        # Get the network components
        data = array([[0, 1], [1, 0]])
        cdata = LabeledCData(data, labels=array([1, 0]))
        encoder = BinaryEncoding(cdata)
        layer1 = ProductAnsatz(2)
        layer2 = ProductAnsatz(2)
        measure = Measurement(2, [0])

        # Make the network
        qnn = Network([encoder, layer1, layer2, measure], "2q-qvm")

        # Build a circuit for the network
        net0 = qnn._build(0)

        # Checks
        self.assertEqual(type(net0), BaseAnsatz)

    def test_get_item(self):
        """Tests getting the correct circuit."""
        # Get network components
        data = array([[0], [1]])
        cdata = LabeledCData(data, labels=array([0, 1]))
        encoder = BinaryEncoding(cdata)
        unitary = ProductAnsatz(1)

        # Make the network
        qnn = Network([encoder, unitary, Measurement(1, [0])], "1q-qvm")

        # Checks
        self.assertEqual(type(qnn[0]), BaseAnsatz)

    @staticmethod
    def get_test_network(computer):
        """Returns a 'test network' to be used in test cases. Utility function."""
        # get network components
        data = array([[0], [1]])
        cdata = LabeledCData(data, labels=array([0, 1]))
        encoder = BinaryEncoding(cdata)
        unitary = ProductAnsatz(1)
        measure = Measurement(1, [0])
        return Network([encoder, unitary, measure], computer)

    def test_computer_string(self):
        """Tests storing the computer as a backend when a string is given."""
        # Computer to use for the network
        comp = "1q-qvm"

        # Get a network with the computer
        qnn = self.get_test_network(comp)

        # Checks
        self.assertEqual(type(qnn.computer), QuantumComputer)

    def test_compile(self):
        """Tests compiling a network for all data points."""
        # Get a network
        qnn = self.get_test_network("1q-qvm")

        # Compile a data point
        executable = qnn.compile(index=0, shots=1000)

        # Checks
        self.assertEqual(type(executable), Program)

    def test_propagate(self):
        """Tests propagating a data point through a network."""
        # Get network components
        data = array([[0], [1]])
        cdata = LabeledCData(data, labels=array([0, 1]))
        encoder = BinaryEncoding(cdata)
        unitary = ProductAnsatz(1)
        measure = Measurement(1, [0])
        qnn = Network([encoder, unitary, measure], "1q-qvm")

        # Propagate the zeroth data point
        out = qnn.propagate(0, shots=10)

        print(out)

    def test_propagate_with_angles(self):
        """Tests propagating a data point through a network with specified
        angles for the ansatz.
        """
        # Get network components
        data = array([[0], [1]])
        cdata = LabeledCData(data, labels=array([0, 1]))
        encoder = BinaryEncoding(cdata)
        ansatz = ProductAnsatz(1)
        measure = Measurement(1, [0])

        # Make the network
        qnn = Network([encoder, ansatz, measure], "1q-qvm")

        print(qnn._ansatz.params._values)

        # Get angles to propagate with
        angles = {0: [1.0, 0.0, 0.0]}

        # Propagate the network
        out = qnn.propagate(0, angles, shots=10)

        print(qnn._ansatz.params._values)

        print(out)

    def test_predict(self):
        """Tests getting a prediction for a data point propagated through the network."""
        # Get components for the network
        data = array([[1, 0], [0, 1]])
        cdata = LabeledCData(data, labels=array([0, 1]))
        encoder = BinaryEncoding(cdata)
        ansatz = ProductAnsatz(2)
        measure = Measurement(2, [0, 1])

        # Define a basic predictor (function which inputs a measurement outcome and returns a label)
        def predictor(outcome):
            return 0

        # Build the network
        qnn = Network(layers=[encoder, ansatz, measure], computer="2q-qvm", predictor=predictor)

        # Get the prediction for each data point
        predict1 = qnn.predict(0)
        predict2 = qnn.predict(1)

        # Make sure the predictions are both correct
        self.assertEqual(predict1, 0)
        self.assertEqual(predict2, 0)

    def test_cost_of_point(self):
        """Tests Network.cost_of_point."""
        # Get components for the network
        data = array([[1, 0], [0, 1]])
        cdata = LabeledCData(data, labels=array([0, 1]))
        encoder = BinaryEncoding(cdata)
        ansatz = ProductAnsatz(2)
        measure = Measurement(2, [0, 1])

        # Define a basic predictor (function which inputs a measurement outcome and returns a label)
        def predictor(outcome):
            return 0

        # Build the network
        qnn = Network(layers=[encoder, ansatz, measure], computer="2q-qvm", predictor=predictor)

        # Compute the cost of each point
        cost0 = qnn.cost_of_point(index=0)
        cost1 = qnn.cost_of_point(index=1)

        # Make sure the costs are correct
        self.assertEqual(cost0, 0)
        self.assertEqual(cost1, 1)

    def test_cost(self):
        """Tests that Network.cost returns a correct value for a given Network.."""
        # Get components for the network
        data = array([[1, 0], [0, 1]])
        cdata = LabeledCData(data, labels=array([0, 1]))
        encoder = BinaryEncoding(cdata)
        ansatz = ProductAnsatz(2)
        measure = Measurement(2, [0, 1])

        # Define a basic predictor (function which inputs a measurement outcome and returns a label)
        def predictor(outcome):
            return 0

        # Build the network
        qnn = Network(layers=[encoder, ansatz, measure], computer="2q-qvm", predictor=predictor)

        # Compute the cost of the network
        cost = qnn.cost(angles={0: [0.0], 1: [0.0]})

        self.assertAlmostEqual(cost, 0.5)

    def test_train(self):
        # Get components for the network
        data = array([[1], [1]])
        cdata = LabeledCData(data, labels=array([1, 1]))
        encoder = BinaryEncoding(cdata)
        ansatz = ProductAnsatz(1)
        measure = Measurement(1, [0])

        # Define a basic predictor (function which inputs a measurement outcome and returns a label)
        def predictor(outcome):
            return 1

        # Build the network
        qnn = Network(layers=[encoder, ansatz, measure], computer="1q-qvm", predictor=predictor)

        # Compute the cost of the network. It should be zero.
        cost = qnn.cost(angles={0: [0.0]})

        # Train the network
        res = qnn.train(trainer="COBYLA", initial_angles=[0.0])

        print(res)


if __name__ == "__main__":
    unittest.main()
