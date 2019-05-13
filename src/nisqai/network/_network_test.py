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


def test_simple():
    """Tests if a Network can be instantiated."""
    # get components for a simple network
    cdata = random_data(num_features=2, num_samples=100, labels=[1 if x < 50 else 0 for x in range(100)])
    encoder = DenseAngleEncoding(cdata, angle_simple_linear, nearest_neighbor(2, 1))
    layer = ProductAnsatz(1)
    measure = Measurement(1, range(1))
    computer = "1q-qvm"

    # make the network
    qnn = Network([encoder, layer, measure], computer)

    # check some basics
    print(type(qnn.computer))
    print(get_qc(computer))


def test_build_basic():
    """Tests building a simple Network."""
    # get the components for a network
    data = array([[0, 1], [1, 0]])
    cdata = CData(data)
    encoder = BinaryEncoding(cdata)
    layer = ProductAnsatz(2)
    measure = Measurement(2, [0])

    # make the network
    qnn = Network([encoder, layer, measure], computer="2q-qvm")

    # build the network
    net0 = qnn._build(0)
    net1 = qnn._build(1)

    # checks
    assert type(net0) == BaseAnsatz
    assert type(net1) == BaseAnsatz


def test_build_multiple_ansatze():
    """Tests building a Network with multiple sequential unitary ansatze."""
    # TODO: options for adding ansatze together -- keep parameters the same or define new ones?
    # get the network components
    data = array([[0, 1], [1, 0]])
    cdata = LabeledCData(data, labels=array([1, 0]))
    encoder = BinaryEncoding(cdata)
    layer1 = ProductAnsatz(2)
    layer2 = ProductAnsatz(2)
    measure = Measurement(2, [0])

    # build the network
    qnn = Network([encoder, layer1, layer2, measure], "2q-qvm")

    # build the network
    net0 = qnn._build(0)

    # checks
    assert type(net0) == BaseAnsatz


def test_get_item():
    """Tests getting the correct circuit."""
    # get network components
    data = array([[0], [1]])
    cdata = LabeledCData(data, labels=array([0, 1]))
    encoder = BinaryEncoding(cdata)
    unitary = ProductAnsatz(1)

    # make the network
    qnn = Network([encoder, unitary, Measurement(1, [0])], "1q-qvm")

    # checks
    assert type(qnn[0] == BaseAnsatz)


def get_test_network(computer):
    """Returns a 'test network' to be used in test cases. Utility function."""
    # get network components
    data = array([[0], [1]])
    cdata = LabeledCData(data, labels=array([0, 1]))
    encoder = BinaryEncoding(cdata)
    unitary = ProductAnsatz(1)
    measure = Measurement(1, [0])
    return Network([encoder, unitary, measure], computer)


def test_computer_string():
    """Tests storing the computer as a backend when a string is given."""
    # computer to use for the network
    comp = "1q-qvm"

    # get a network with the computer
    qnn = get_test_network(comp)

    # checks
    assert type(qnn.computer) == QuantumComputer


def test_compile():
    """Tests compiling a network for all data points."""
    # get a network
    qnn = get_test_network("1q-qvm")

    # compile a data point
    executable = qnn.compile(index=0, shots=1000)

    # checks
    assert type(executable) == Program


def test_propagate():
    """Tests propagating a data point through a network."""
    # get network components
    data = array([[0], [1]])
    cdata = LabeledCData(data, labels=array([0, 1]))
    encoder = BinaryEncoding(cdata)
    unitary = ProductAnsatz(1)
    measure = Measurement(1, [0])
    qnn = Network([encoder, unitary, measure], "1q-qvm")

    print(qnn[0])

    # propagate the zeroth data point
    out = qnn.propagate(0, shots=10)

    print(out)


def test_propagate_with_angles():
    """Tests propagating a data point through a network with specified
    angles for the ansatz.
    """
    # get network components
    data = array([[0], [1]])
    cdata = LabeledCData(data, labels=array([0, 1]))
    encoder = BinaryEncoding(cdata)
    ansatz = ProductAnsatz(1)
    measure = Measurement(1, [0])

    # make the network
    qnn = Network([encoder, ansatz, measure], "1q-qvm")

    print(qnn._ansatz.params._values)

    # get angles to propagate with
    angles = {0: [1.0, 0.0, 0.0]}

    # propagate the network
    out = qnn.propagate(0, angles, shots=10)

    print(qnn._ansatz.params._values)

    print(out)


def test_predict():
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
    assert(predict1 == 0)
    assert(predict2 == 0)


def test_cost_of_point():
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
    assert cost0 == 0
    assert cost1 == 1


def test_cost():
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

    assert abs(cost - 0.5) <= 1e-3


def test_train():
    # get network components
    data = array([[0], [1]])
    cdata = LabeledCData(data, labels=array([0, 1]))
    encoder = BinaryEncoding(cdata)
    unitary = ProductAnsatz(1)
    measure = Measurement(1, [0])
    qnn = Network([encoder, unitary, measure], "1q-qvm")

    trainout = qnn.train(10)

    print("trainout =", trainout)


if __name__ == "__main__":
    test_simple()
    test_build_basic()
    test_build_multiple_ansatze()
    test_get_item()
    test_computer_string()
    test_compile()
    test_propagate()
    test_propagate_with_angles()
    test_predict()
    test_cost_of_point()
    test_cost()
    #test_train()
    print("All tests for Network passed.")
