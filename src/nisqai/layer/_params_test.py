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

# Imports
import unittest

from pyquil import Program
from pyquil.quilatom import MemoryReference

from nisqai.layer._params import Parameters, product_ansatz_parameters


class ParametersTest(unittest.TestCase):
    """Unit tests for Parameters class."""

    def test_basic(self):
        """Tests that a parameter class can be instantiated."""
        # Dictionary of arameter values
        params = {0: [0, 1],
                  1: [2, 3, 4]}

        # Create an instance of Parameters
        parameters = Parameters(params)

        # Make sure the class was instantiated correctly
        self.assertEqual(parameters.values, params)

    def test_no_parameters_on_qubit(self):
        """Tests that a Parameters class with no parameters on a
        particular qubit can be created.
        """
        # create a Parameters class
        params = Parameters(
            {0: [1, 2, 3],
             1: [4, 5, 6],
             2: []}
        )

        # Make sure the names attribute stores the qubit with no parameters
        self.assertIn(2, params.names.keys())

        # Test if the list of values is correct
        self.assertEqual(params.list_values(), [1, 2, 3, 4, 5, 6])

    def test_names(self):
        """Tests that the parameter names is correct."""
        # Create an instance of Parameters
        params = Parameters(
            {0: [1, 0],
             1: [4],
             2: []}
        )

        # Define the correct names
        correct_names = {0: ["q_000_g_000", "q_000_g_001"],
                         1: ["q_001_g_000"],
                         2: []
                         }

        # Test if the names are correct
        self.assertEqual(params.names, correct_names)

    def test_names_multi_digits(self):
        """Tests the naming convention is correct for large numbers
        of gates and qubits (two and three digits).
        """
        # create a Parameters class with two and three digit gate indices
        params = Parameters(
            {0: list(range(150)),
             1: [2, 3, 4]}
        )

        # Make sure some two digit gate indices are correct
        self.assertIn("q_000_g_015", params.list_names())
        self.assertIn("q_000_g_099", params.list_names())

        # Make sure some three digit gate indices are correct
        self.assertIn("q_000_g_123", params.list_names())
        self.assertIn("q_000_g_100", params.list_names())

    def test_grid_names(self):
        """Tests that grid names appear properly for Parameters."""
        # create a Parameters class
        params = Parameters(
            {0: [1, 2],
             1: [3, 4]}
        )

        # correct grid names output
        correct = [['q_000_g_000', 'q_000_g_001'], ['q_001_g_000', 'q_001_g_001']]

        # test if the grid names are correct
        self.assertEqual(params.grid_names(), correct)

    def test_grid_values(self):
        """Tests that grid values appear properly for Parameters."""
        # create a Parameters class
        params = Parameters(
            {0: [1, 2],
             1: [3, 4]}
        )

        # correct grid values output
        correct = [[1, 2], [3, 4]]

        # test if the grid names are correct
        self.assertEqual(params.grid_values(), correct)

    def test_unique_names(self):
        """Creates multiple instances of Parameters and checks that
        the Parameters.names are all unique.
        """
        # Create a Parameters class with 18 parameters
        params1 = Parameters(
            {0: list(range(15)),
             1: [2, 3, 4]}
        )

        # Make sure the names are unique
        self.assertEqual(len(set(params1.list_names())), 18)

        # Create a Parameters class with 1 parameter
        params2 = Parameters(
            {0: [0],
             1: [],
             2: [],
             3: []}
        )

        # Make sure the names are unique
        self.assertEqual(len(set(params2.list_values())), 1)

        # Create a Parameters class with 999 parameters (!)
        params3 = Parameters(
            {0: list(range(999))}
        )

        # Make sure the names are unique
        self.assertEqual(len(set(params3.list_names())), 999)

    def test_depth(self):
        """Tests the depth is computed correctly for multiple Parameters."""
        # Create parameters
        param1 = Parameters(
            {0: list(range(100)),
             1: [2, 3, 4]}
        )

        # Test the depth is correct
        self.assertEqual(param1.depth(), 100)

        # Create another parameters object
        param2 = Parameters(
            {0: [1, 2, 3],
             1: [4, 5, 6],
             2: [],
             3: [1, 2, 2, 2, 1]}
        )

        # Test the depth is correct
        self.assertEqual(param2.depth(), 5)

    def test_memory_map(self):
        """Tests that a memory map is correct."""
        # parameter value dictionary
        params = {0: [0, 1],
                  1: [2, 3, 4]}

        # create a Parameters class
        parameters = Parameters(params)

        # define the correct memory map
        correct_map = {"q_000_g_000": [0.0],
                       "q_000_g_001": [1.0],
                       "q_001_g_000": [2.0],
                       "q_001_g_001": [3.0],
                       "q_001_g_002": [4.0]}

        # test if the memory map is correct
        self.assertEqual(parameters.memory_map(), correct_map)

    def test_memory_reference(self):
        """Simple test for memory references."""
        # program to declare memory references for
        prog = Program()

        # get some Parameters
        params = Parameters(
            {0: [1, 2],
             1: [3, 4]}
        )

        # declare the memory references for all parameters in the program
        params.declare_memory_references(prog)

        # test for correctness
        self.assertEqual(len(params.memory_references.keys()), 2)

        # Make sure the memory references are correct types
        for mref in list(params.memory_references.values()):
            self.assertEqual(type(mref[0]), MemoryReference)
            self.assertEqual(type(mref[1]), MemoryReference)

    def test_memory_reference_indexing(self):
        """Tests indexing for memory references."""
        # program to declare memory references for
        prog = Program()

        # get some Parameters
        params = Parameters(
            {0: [1, 2],
             1: [3, 4]}
        )

        # declare the memory references for all parameters in the program
        params.declare_memory_references(prog)

        # test for correctness
        self.assertEqual(params.memory_references[0][0].name, "q_000_g_000")
        self.assertEqual(params.memory_references[0][1].name, "q_000_g_001")
        self.assertEqual(params.memory_references[1][0].name, "q_001_g_000")
        self.assertEqual(params.memory_references[1][1].name, "q_001_g_001")

    def test_product_ansatz_parameters(self):
        """Tests getting parameters for the ProductAnsatz class."""
        # Get the product ansatz parameters
        params = product_ansatz_parameters(3, 2, 0.0)

        # define the correct param values
        correct = {0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0]}

        # Test if the params are correct
        self.assertEqual(params.values, correct)

    def test_update_parameters(self):
        """Tests for correct values after updating parameters."""
        # get some Parameters
        params = Parameters({0: [0]})

        # ensure the initial memory map is correct
        self.assertEqual(params.memory_map(), {"q_000_g_000": [0.0]})

        # update the values
        params.update_values({0: [1]})

        # ensure the new memory map is correct
        self.assertEqual(params.memory_map(), {"q_000_g_000": [1.0]})

    def test_update_parameters_memory_map(self):
        """Tests updating parameters and returning a memory map."""
        # get some Parameters
        params = Parameters({0: [0]})

        # ensure the initial memory map is correct
        self.assertEqual(params.memory_map(), {"q_000_g_000": [0.0]})

        new_map = params.update_values_memory_map({0: [1]})

        self.assertEqual(params.values, {0: [1.0]})
        self.assertEqual(new_map, {"q_000_g_000": [1.0]})

    def test_list_to_dict(self):
        """Tests converting a list to a dictionary for the parameters."""
        # Get a parameters object
        params = Parameters({0: [1, 2, 3],
                             1: [None, 4, None],
                             2: [5, 6, None]})

        # Make a list of new parameter values of the correct length
        nqubits, depth = params.shape()
        list_values = list(range(nqubits * depth))

        # Turn the list into a dictionary of values
        dict_values = params._list_to_dict(list_values)

        # Define the correct dictionary
        correct = {0: [0, 1, 2],
                   1: [3, 4, 5],
                   2: [6, 7, 8]}

        # Make sure the converted list is correct
        self.assertEqual(dict_values, correct)

    def test_update_parameters_with_list(self):
        """Tests updating parameters using a list of new parameter values."""
        # Get some Parameters
        params = Parameters({0: [0]})

        # Ensure the initial memory map is correct
        self.assertEqual(params.memory_map(), {"q_000_g_000": [0.0]})

        # Update the values
        params.update_values([1])

        # Ensure the new memory map is correct
        self.assertEqual(params.memory_map(), {"q_000_g_000": [1.0]})

    def test_update_parameters_memory_map_with_list(self):
        """Tests updating parameters and returning a memory map with a list of new
        parameter values.
        """
        # get some Parameters
        params = Parameters({0: [0]})

        # Ensure the initial memory map is correct
        self.assertEqual(params.memory_map(), {"q_000_g_000": [0.0]})

        # Update the values
        memory_map = params.update_values_memory_map([1])

        # Ensure the returned memory map is correct
        self.assertEqual(memory_map, {"q_000_g_000": [1.0]})

        # Ensure the new memory map is correct
        self.assertEqual(params.memory_map(), {"q_000_g_000": [1.0]})


if __name__ == "__main__":
    unittest.main()
