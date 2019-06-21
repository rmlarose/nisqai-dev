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

from numpy import array, ndarray
import unittest

from pyquil import Program, get_qc
from pyquil.gates import X, MEASURE

from nisqai.measure._measurement_outcome import MeasurementOutcome


class TestMeasuremnetOutcome(unittest.TestCase):
    @staticmethod
    def get_all_zeros_outcome(nqubits, nshots):
        """Helper function that returns the outcome of all zeros.

        Args:
            nqubits : int
                Number of qubits in the circuit.

            nshots : int
                Number of shots to simulate the circuit.
        """
        prog = Program()
        creg = prog.declare("ro", memory_type="BIT", memory_size=nqubits)

        prog += [MEASURE(q, creg[q]) for q in range(nqubits)]

        prog.wrap_in_numshots_loop(nshots)

        computer = get_qc("{}q-qvm".format(nqubits))

        return computer.run(prog)

    @staticmethod
    def get_all_ones_outcome(nqubits, nshots):
        """Helper function that returns the outcome of all ones.

        Args:
            nqubits : int
                Number of qubits in the circuit.

            nshots : int
                Number of shots to simulate the circuit.
        """
        prog = Program()
        creg = prog.declare("ro", memory_type="BIT", memory_size=nqubits)

        prog += [X(q) for q in range(nqubits)]
        prog += [MEASURE(q, creg[q]) for q in range(nqubits)]

        prog.wrap_in_numshots_loop(nshots)

        computer = get_qc("{}q-qvm".format(nqubits))

        return computer.run(prog)

    def test_basic(self):
        """Tests that a MeasurementOutcome can be instantiated."""
        # get an outcome from simulating a circuit
        result = self.get_all_ones_outcome(4, 10)

        # create a MeasurementOutcome
        outcome = MeasurementOutcome(result)

        # trivial check
        self.assertTrue((outcome.raw_outcome == result).all())

    def test_num_qubits(self):
        """Tests that a MeasurementOutcome has the right qubit number."""
        # number of qubits
        nqubits = 4

        # get an outcome from simulating a circuit
        result = self.get_all_ones_outcome(nqubits, 10)

        # create a MeasurementOutcome
        outcome = MeasurementOutcome(result)

        # trivial check
        self.assertEqual(outcome.num_qubits, nqubits)

    def test_num_shots(self):
        """Tests that a MeasurementOutcome has the right number of shots."""
        # number of qubits
        nqubits = 4

        # number of shots
        nshots = 40

        # get an outcome from simulating a circuit
        result = self.get_all_ones_outcome(nqubits, nshots)

        # create a MeasurementOutcome
        outcome = MeasurementOutcome(result)

        # trivial check
        self.assertEqual(outcome.shots, nshots)

    def test_get_item(self):
        """Tests getting an item from a measurement outcome."""
        # number of qubits
        nqubits = 5

        # number of shots
        nshots = 40

        # get an outcome from simulating a circuit
        result = self.get_all_ones_outcome(nqubits, nshots)

        # create a MeasurementOutcome
        outcome = MeasurementOutcome(result)

        self.assertEqual(len(outcome[0]), 5)

    def test_len(self):
        """Tests the length of a measurement outcome."""
        # get an outcome from simulating a circuit
        result = self.get_all_ones_outcome(nqubits=2, nshots=1000)

        # create a MeasurementOutcome
        outcome = MeasurementOutcome(result)

        self.assertEqual(len(outcome), 1000)

    def test_as_int(self):
        """Tests the integer value of bit strings is correct."""
        # get some measurement outcomes
        zeros = MeasurementOutcome(self.get_all_zeros_outcome(nqubits=2, nshots=20))
        ones = MeasurementOutcome(self.get_all_ones_outcome(nqubits=2, nshots=20))

        # checks for zeros
        self.assertTrue(type(zeros.as_int(0)), int)
        self.assertEqual(zeros.as_int(0), 0)

        # checks for ones
        self.assertTrue(type(ones.as_int(0)), int)
        self.assertEqual(ones.as_int(0), 3)

    def test_as_int_big_int(self):
        """Tests the integer value of bit strings for large integers."""
        # get a measurement outcome
        ones = MeasurementOutcome(self.get_all_ones_outcome(nqubits=10, nshots=20))

        # checks for ones
        self.assertTrue(type(ones.as_int(0)), int)
        self.assertEqual(ones.as_int(0), 2**10 - 1)

    def test_average_all_zeros(self):
        """Tests the average outcome of all zero measurements is all zeros."""
        # Get an all zero MeasurementOutcome
        zeros = MeasurementOutcome(self.get_all_zeros_outcome(nqubits=4, nshots=20))

        # Compute the average
        avg = zeros.average()

        # Make sure it's all zeros
        self.assertTrue(type(avg) == ndarray)
        self.assertEqual(len(avg), zeros.num_qubits)
        self.assertTrue(sum(avg) == 0)

    def test_average_all_ones(self):
        """Tests the average outcome of all ones measurements is all ones."""
        # Get an all zero MeasurementOutcome
        ones = MeasurementOutcome(self.get_all_ones_outcome(nqubits=4, nshots=20))

        # Compute the average
        avg = ones.average()

        # Make sure it's all zeros
        self.assertTrue(type(avg) == ndarray)
        self.assertEqual(len(avg), ones.num_qubits)
        self.assertTrue(sum(avg) == ones.num_qubits)

    def test_average(self):
        """Tests that the average is computed correctly for a given raw outcome."""
        # Example result
        result = array([[1, 0], [0, 1]])

        # Make a MeasurementOutcome
        meas = MeasurementOutcome(result)

        # Compute the average
        avg = meas.average()

        # Make sure its correct
        self.assertAlmostEqual(avg[0], 0.5)
        self.assertAlmostEqual(avg[1], 0.5)


if __name__ == "__main__":
    unittest.main()
