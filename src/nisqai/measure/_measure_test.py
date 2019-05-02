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

#from nisqai.measure._measure import Measurement
from nisqai.measure._measure import Measurement, measure_all, measure_qubit, measure_top
import numpy as np
import unittest

class TestInvalidInputs(unittest.TestCase):
    '''Testing for ValueError and TypeError exceptions'''

    def test_invalid_qubit_number(self):
        """Tests if a negative integer is inputted for number of qubits."""
        with self.assertRaises(ValueError):
            Measurement(-1, range(9))

    def test_invalid_qubit_number_bell_meas(self):
        """Tests if a Bell basis measurement is attempted on more than 2 qubits."""

        bell_basis = ['Bell', None]
        with self.assertRaises(ValueError):
            Measurement(4, [0,1,2,3], bell_basis[0], bell_basis[1])

    def test_invalid_type_basis(self):
        """Tests if an invalid type is entered for a change of basis parameters."""

        measure = Measurement(4, [0,1,2,3])
        new_basis_float = 3.0
        
        with self.assertRaises(TypeError):
            measure.change_basis(new_basis_float)

    def test_more_qubits_than_available(self):
        '''Tests if a measurement is attempted on an un-available qubit.'''
        with self.assertRaises(ValueError):
            measure_qubit(3, 7)


def test_measure_all():
    """Tests measurements on all qubits."""
    measure = Measurement(9, range(9))
    assert measure.num_qubits == 9
    assert measure.num_measurements == 9
    assert measure.measured_qubits == list(range(9))
    assert len(measure.circuit.instructions) == 10

    measure_all_qbs = measure_all(3)
    assert measure_all_qbs.num_qubits == 3
    assert measure_all_qbs.num_measurements == 3
    assert measure_all_qbs.measured_qubits == [0, 1, 2]

def test_measure_single():
    """Tests measurements on a single qubit."""
    measure = Measurement(10, [1])
    assert measure.num_qubits == 10
    assert measure.num_measurements == 1
    assert measure.measured_qubits == [1]
    assert len(measure.circuit.instructions) == 2


    measure_top_qubit = measure_top(5)
    assert measure_top_qubit.num_qubits == 5
    assert measure_top_qubit.measured_qubits == [0]

    measure_single_qubit = measure_qubit(5, 2)
    assert measure_single_qubit.num_qubits == 5
    assert measure_single_qubit.measured_qubits == [2]

def test_measure_some():
    """Tests measurements on a subset of qubits."""
    measure = Measurement(10, [3, 5, 8])
    assert measure.num_qubits == 10
    assert measure.num_measurements == 3
    assert measure.measured_qubits == [3, 5, 8]


def test_measure_tuple():
    """Tests measurements on all qubits with a tuple as input."""
    measure = Measurement(4, (0, 1, 2, 3))
    assert measure.num_qubits == 4
    assert measure.measured_qubits == [0, 1, 2, 3]
    assert measure.num_measurements == 4


def test_measure_change_basis():
    """Tests changing basis of a measurement.
    (Changes basis measurement to a RZ(pi/4) and GHZ basis ~ 1/sqrt(2)[|000..0> +|111..1>]
    """
    # bases to use
    old_basis = None
  
    new_basis_rz = ['RZ', np.pi / 4]

    new_basis_rx = ['RX', np.pi / 8]

    new_basis_ry = ['RY', np.pi / 24]

    new_basis_ghz = 'GHZ'

    new_basis_h = 'H'

    new_basis_bell = 'Bell'

    new_basis_xy = ['XY', np.pi / 16]


    # measure in first (old) basis
    measure = Measurement(2, range(2))

    assert measure.num_qubits == 2
    assert measure.num_measurements == 2
    assert measure.measured_qubits == [0, 1]

    # change basis to RZ(pi/4)
    measure.change_basis(new_basis_rz)
    
    assert measure.basis_gate == 'RZ'
    assert measure.basis_angle == np.pi/4

    # change basis to RX(pi/8)
    measure.change_basis(new_basis_rx)
    
    assert measure.basis_gate == 'RX'
    assert measure.basis_angle == np.pi/8

    # change basis to RY(pi/24)
    measure.change_basis(new_basis_ry)
    
    assert measure.basis_gate == 'RY'
    assert measure.basis_angle == np.pi/24

    # change basis to GHZ
    measure.change_basis(new_basis_ghz)

    assert measure.basis_gate == 'GHZ'
    assert measure.basis_angle == None

    # change basis to |+/->
    measure.change_basis(new_basis_h)

    assert measure.basis_gate == 'H'
    assert measure.basis_angle == None

    # change basis to Bell basis
    measure.change_basis(new_basis_bell)

    assert measure.basis_gate == 'Bell'
    assert measure.basis_angle == None

    # change basis to x/y plane
    measure.change_basis(new_basis_xy)
    assert measure.basis_gate == 'XY'
    assert measure.basis_angle ==  np.pi / 16
    assert len(measure.circuit.instructions) == 7


if __name__ == "__main__":
    test_measure_all()
    test_measure_single()
    test_measure_some()
    test_measure_tuple()
    test_measure_change_basis()
    unittest.main()
    print("All tests for Measurement passed.")

