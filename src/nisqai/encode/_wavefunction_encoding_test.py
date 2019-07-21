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

import unittest

from nisqai.data import CData
from nisqai.encode import WaveFunctionEncoding

from numpy import allclose, array, random, zeros
from numpy.linalg import norm

from pyquil.gates import SWAP
from pyquil.api import WavefunctionSimulator


class WaveFunctionEncodingTest(unittest.TestCase):
    """Unit tests for WaveFunctionEncoding."""

    # TODO: Make this unit test pass.
    def test_input_output_comparison(self):
        # Number of qubits
        nqubits = 2

        # Number of data points
        npoints = 10

        # Wave function simulator for checking the output of the circuit
        sim = WavefunctionSimulator()

        # Empty list to build data
        data = []

        # Use ten data points
        for _ in range(npoints):
            # Get a random complex feature vector
            feature_vector = random.normal(size=2**nqubits) + 1j*random.normal(size=2**nqubits)

            # Append this to the data
            data.append(feature_vector)

        # Get a CData object
        cdata = CData(array(data))

        # Get the wavefunction encoding for the cdata
        encoder = WaveFunctionEncoding(cdata)

        # Get the zero vector for help with comparisons
        zero = zeros(2**nqubits)
        zero[0] = 1

        # Loop over all encoder circuits
        for ii in range(npoints):
            # Grab the current encoder circuit
            ansatz = encoder[ii]

            # Add a SWAP gate to get the original ordering
            ansatz.circuit.inst(SWAP(0, 1))

            # Get the wavefunction of the circuit
            wavefunction = sim.wavefunction(ansatz.circuit)

            actual = cdata.data[ii] / norm(cdata.data[ii], ord=2)
            computed = wavefunction.amplitudes

            # Make sure it's close to the input feature vector
            self.assertTrue(allclose(computed, actual))


if __name__ == "__main__":
    unittest.main()
