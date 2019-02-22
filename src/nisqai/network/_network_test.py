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

from pyquil import Program
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
    assert qnn.computer == computer


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


def test_backend():
    """Tests that the backend is correct for a Network."""
    # computer to use for the network
    comp = "1q-qvm"

    # get a network with the computer
    qnn = get_test_network(comp)

    # checks
    assert type(qnn.backend) == QuantumComputer
    assert qnn.computer == comp


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
    # get a network
    qnn = get_test_network("1q-qvm")

    print(qnn[0])

    # propagate the zeroth data point
    out = qnn.propagate(1, None, shots=10)

    print(out)


if __name__ == "__main__":
    test_simple()
    test_build_basic()
    test_build_multiple_ansatze()
    test_get_item()
    test_backend()
    test_compile()
    test_propagate()
    print("All tests for Network passed.")
