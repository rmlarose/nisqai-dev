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


from nisqai.utils import startQVMandQUILC, stopQVMandQUILC
from nisqai.cost._quantum_costs import Observable
import numpy as np
import unittest
import pyquil


class TestObservable(unittest.TestCase):
    '''Testing for ValueError and TypeError exceptions'''

    def test_invalid_qubit_number(self):
        """Tests if a negative integer is inputted for number of qubits."""
        with self.assertRaises(ValueError):
            Observable(-1, range(9))

    def test_invalid_shot_number(self):
        """Tests if a valid number of measurement shots is inputted"""
        
        obs = Observable(4, [0,1,2])
        with self.assertRaises(TypeError):
            obs.compute_observables_exp('4q-qvm', 'num_shots', 'Z')
        with self.assertRaises(ValueError):
            obs.compute_observables_exp('4q-qvm', -20, 'Z')

    def test_observble(self):
        observable_1 = Observable(3, [0, 1])
        self.assertEqual(observable_1.num_qubits, 3)
        self.assertEqual(observable_1.measured_qubits, [0, 1])

        observable_2 = Observable(5, [2, 3, 4])
        self.assertEqual(observable_2.num_qubits, 5)
        self.assertEqual(observable_2.measured_qubits, [2, 3, 4])

    def test_add_single_qubit_observable_meas(self):
        """Test if correct circuit has been generated to measure required observables"""
        # Check X observable
        observable_1 = Observable(3, [0, 1])
        observable_1.add_single_qubit_observable_meas('X')
        # Check if first instruction is declare statement
        self.assertEqual(observable_1.circuit.instructions[0].name.lower(), 'ro')
        for qubit in [0, 1]:
            self.assertEqual(observable_1.circuit.instructions[1+qubit].name.lower(), 'h')
            self.assertEqual(observable_1.circuit.instructions[1+qubit].qubits[0].index, qubit)
        
        # Check Z observable
        observable_2 = Observable(3, [0, 1, 2])
        observable_2.add_single_qubit_observable_meas('Z')
        self.assertEqual(observable_2.circuit.instructions[0].name.lower(), 'ro')

        # For a Z observable measurement, there should be no intermediate 
        # gates between DECLARE & MEASURE
        self.assertIsInstance(observable_2.circuit.instructions[1], pyquil.quilbase.Measurement)
        
        # Check Y observable
        observable_4 = Observable(5, [0])
        observable_4.add_single_qubit_observable_meas('Y')
        self.assertEqual(observable_4.circuit.instructions[1].name.lower(), 'phase')
        self.assertEqual(observable_4.circuit.instructions[1].params[0], np.pi/2)

        self.assertEqual(observable_4.circuit.instructions[2].name.lower(), 'h')

    def test_compute_observables_exp(self):
        """Test if expectation value of observables are computed correctly"""
        observable_1 = Observable(3, [0, 1])
        exp_values_1 = observable_1.compute_observables_exp('3q-qvm', 100000, 'X')
        for expectation_value in exp_values_1:
            self.assertAlmostEqual(abs(expectation_value), 0, 1)

        observable_2 = Observable(4, [0, 1, 2])
        exp_values_2 = observable_2.compute_observables_exp('4q-qvm', 100000, 'Z')
        for expectation_value in exp_values_2:
            self.assertAlmostEqual(abs(expectation_value), 1, 5)

        observable_3 = Observable(1, [0])
        exp_values_3 = observable_3.compute_observables_exp('1q-qvm', 100000, 'Y')
        for expectation_value in exp_values_3:
            self.assertAlmostEqual(abs(expectation_value), 0, 1)

    def test_sum_observables(self):
        """Test if sum of expectation values of observables is computed correctly"""
        observable_1 = Observable(3, [0, 1])
        sum_observables_1 = observable_1.sum_observables('3q-qvm', 100000, 'X')
        self.assertAlmostEqual(abs(sum_observables_1), 0, 1)

        observable_2 = Observable(2, [0, 1])
        sum_observables_2 = observable_2.sum_observables('2q-qvm', 100000, 'Z')
        self.assertAlmostEqual(abs(sum_observables_2), 2, 5)


if __name__ == "__main__":
    # Start the Rigetti QVM and Quil compiler
    qvm_server, quilc_server, _ = startQVMandQUILC()

    # Do the unit tests
    unittest.main()

    # Stop the Rigetti QVM and Quil compiler
    stopQVMandQUILC(qvm_server, quilc_server)
