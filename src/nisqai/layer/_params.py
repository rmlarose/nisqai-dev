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

from itertools import chain

# Standard specification format for strings.
# Fill character: 0
# Right justified: >
# Three digits long: 3
FORMAT_SPEC = "0>3"


class Parameters:
    """Efficient data structure for storing and working with parameters in an ansatz.

    Key attributes:

        (1) Parameters.list_values
            One dimensional list of parameter values for use in training.

        (2) Parameters.grid_values
            Two dimensional array of parameter values for efficient placement
            and visualization.

        (3) Parameters.memory_map
            Dictionary of {parameter name: parameter value} pairs for use in
            parameteric compilation.
    """

    def __init__(self, parameters):
        """Initializes a Parameters class.

        Args:
            parameters : dict[int, list[float]]
                Dictionary of

                {qubit : list of parameter values for qubit}

                pairs.

                IMPORTANT: All qubit indices must explicitly be included as keys, even if
                some qubits do not have parameterized gates.

                Qubits with no parameterized gates should have empty lists as values
                for that qubit index.

                Examples:

                    parameters = {0 : [1, 2],
                                  1 : [3, 4]}

                        Corresponds to a circuit which looks like:

                        Qubit 0 ----[1]----[2]----
                        Qubit 1 ----[3]----[4]----

                        That is: A circuit with two qubits, 0 and 1, where qubit 0 has
                        parameters 1 and 2 for its first and second parameterized gates,
                        respectively, and qubit 1 has parameters 3 and 4 for its first
                        and second parameterized gates, respectively.

                        Note that other unparameterized gates can appear in the circuit,
                        at any point before, in between, or after parameterized gates.



                    parameters = {0 : [1, 2],
                                  1 : []
                                  2 : [3]}

                        Corresponds to a circuit which looks like:

                        Qubit 0 ----[1]----[2]----
                        Qubit 1 ------------------
                        Qubit 2 ----[3]-----------

                    That is: A circuit with three qubits. Qubit 0 has parameters 1 and 2
                    for its first and second parameterized gates, respectively. Qubit 1
                    has no parameterized gates. Qubit 2 has parameter 3 for its first
                    parameterized gate.
        """
        # store the parameter dictionary
        # TODO: write a method to make sure the parameter dictionary is valid
        self.values = parameters

        # extract the number of qubits
        self._num_qubits = len(self.values.keys())

        # make the dictionary of parameter names
        self.names = self._make_parameter_names()

    def _make_parameter_names(self):
        """Returns a dictionary of names according to the standard naming convention.

        The standard naming convention is given by

        q_ABC_g_XYZ

        where

        ABC = three digit integer label of qubit

        and

        XYZ = three digit integer label of gate.

        Examples:
            q_000_g_005 = Fifth parameterized gate on qubit zero.
            q_999_g_024 = Twenty fourth (!) parameterized gate on qubit 999. (!!!)
        """
        names = {}
        for qubit in self.values.keys():
            names[qubit] = []
            qubit_key = format(qubit, FORMAT_SPEC)
            for gate in range(len(self.values[qubit])):
                gate_key = format(gate, FORMAT_SPEC)
                names[qubit].append(
                    "q_{}_g_{}".format(qubit_key, gate_key)
                )
        return names

    def list_values(self):
        """Returns a one dimensional list of all parameter values."""
        return list(chain.from_iterable(self.values.values()))

    def list_names(self):
        """Returns a one dimensional list of all parameter names."""
        return list(chain.from_iterable(self.names.values()))

    def grid_values(self):
        """Returns a two dimensional array of all parameter values."""
        return list(self.values.values())

    def grid_names(self):
        """Returns a two dimensional array of all parameter names."""
        return list(self.names.values())

    def memory_map(self):
        """Returns a memory map for use in pyQuil.

        A memory map is defined by a dictionary of

        {parameter name: parameter value}

        pairs.
        """
        # TODO: speedup implementation of this method: crucial for fast implementations
        # TODO: make more Pythonic
        mem_map = {}
        for qubit in range(len(self.values)):
            for gate in range(len(self.values[qubit])):
                mem_map[self.names[qubit][gate]] = self.values[qubit][gate]
        return mem_map

    def depth(self):
        """Returns the depth of the Parameters, defined as the maximum length
        of all parameter lists over all qubits.
        """
        return len(max(self.values.values(), key=len))
