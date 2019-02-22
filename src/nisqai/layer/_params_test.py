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

from nisqai.layer._params import Parameters


def test_basic():
    """Tests that a parameter class can be instantiated."""
    # parameter values dictionary
    params = {0: [0, 1],
              1: [2, 3, 4]}

    # create an instance of Parameters
    parameters = Parameters(params)

    # make sure the class was instantiated correctly
    assert parameters.values == params


def test_no_parameters_on_qubit():
    """Tests that a Parameters class with no parameters on a
    particular qubit can be created.
    """
    # create a Parameters class
    params = Parameters(
        {0: [1, 2, 3],
         1: [4, 5, 6],
         2: []}
    )

    # make sure the names attribute stores the qubit with no parameters
    assert 2 in params.names.keys()

    # test if the list of values is correct
    assert params.list_values() == [1, 2, 3, 4, 5, 6]


def test_names():
    """Tests that the parameter names is correct."""
    # create a instance of Parameters
    params = Parameters(
        {0: [1, 0],
         1: [4],
         2: []}
    )

    # define the correct names
    correct_names = {0: ["q_000_g_000", "q_000_g_001"],
                     1: ["q_001_g_000"],
                     2: []
                     }

    # test if the names are correct
    assert params.names == correct_names


def test_names_multi_digits():
    """Tests the naming convention is correct for large numbers
    of gates and qubits (two and three digits).
    """
    # create a Parameters class with two and three digit gate indices
    params = Parameters(
        {0: list(range(150)),
         1: [2, 3, 4]}
    )

    # make sure some two digit gate indices are correct
    assert "q_000_g_015" in params.list_names()
    assert "q_000_g_099" in params.list_names()

    # make sure some three digit gate indices are correct
    assert "q_000_g_123" in params.list_names()
    assert "q_000_g_100" in params.list_names()


def test_unique_names():
    """Creates multiple instances of Parameters and checks that
    the Parameters.names are all unique.
    """
    # create a Parameters class with 18 parameters
    params1 = Parameters(
        {0: list(range(15)),
         1: [2, 3, 4]}
    )

    # make sure the names are unique
    assert len(set(params1.list_names())) == 18

    # create a Parameters class with 1 parameter
    params2 = Parameters(
        {0: [0],
         1: [],
         2: [],
         3: []}
    )

    # make sure the names are unique
    assert len(set(params2.list_values())) == 1

    # create a Parameters class with 999 parameters (!)
    params3 = Parameters(
        {0: list(range(999))}
    )

    # make sure the names are unique
    assert len(set(params3.list_names())) == 999


def test_depth():
    """Tests the depth is computed correctly for multiple Parameters."""
    param1 = Parameters(
        {0: list(range(100)),
         1: [2, 3, 4]}
    )

    assert param1.depth() == 100

    param2 = Parameters(
        {0: [1, 2, 3],
         1: [4, 5, 6],
         2: [],
         3: [1, 2, 2, 2, 1]}
    )

    assert param2.depth() == 5


def test_memory_map():
    """Tests that a memory map is correct."""
    # parameter value dictionary
    params = {0: [0, 1],
              1: [2, 3, 4]}

    # create a Parameters class
    parameters = Parameters(params)

    # define the correct memory map
    correct_map = {"q_000_g_000": 0,
                   "q_000_g_001": 1,
                   "q_001_g_000": 2,
                   "q_001_g_001": 3,
                   "q_001_g_002": 4}

    # test if the memory map is correct
    assert parameters.memory_map() == correct_map


if __name__ == "__main__":
    test_basic()
    test_no_parameters_on_qubit()
    test_names()
    test_names_multi_digits()
    test_unique_names()
    test_depth()
    test_memory_map()
    print("All tests for Parameters class passed.")
