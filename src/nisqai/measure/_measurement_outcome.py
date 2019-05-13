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

"""Class for working with measurement outcomes in pyQuil.

Nominally, pyQuil stores measurement outcomes as nested lists,
with inner lists representing measured bit strings. For example:

[[0, 0], [0, 1], [1, 0], [1, 1]]

represents a circuit sampling two bit strings four times.
The number of inner lists is equal to the number of times
the circuit is run.

A MeasurementOutcome inputs this data and makes it easy to work with,
interpret, visualize, modify, and prepare for future encoding.
"""

from numpy import ndarray


class MeasurementOutcome:
    """Measurement outcome class for dealing with sampled bit strings from circuits."""

    def __init__(self, outcome):
        """Initializes a MeasurementOutcome.

        Args:
            outcome : array-like
                The measurement results obtained from running a circuit.
                For example, the minimal program:
                    from pyquil import Program, get_qc
                    from pyquil.gates import H, MEASURE

                    prog = Program()
                    creg = prog.declare("ro", memory_type="BIT", memory_size=2)

                    prog += [H(0), H(1), MEASURE(0, creg[0]), MEASURE(1, creg[1])]

                    prog.wrap_in_numshots_loop(5)

                    computer = get_qc("2q-qvm")

                    result = computer.run(prog)

                    # example result
                    # type(result) = numpy.ndarray
                    # result = array([[1, 0],
                    #                 [0, 0],
                    #                 [1, 0],
                    #                 [0, 1],
                    #                 [0, 0]])
            """
        # input checking
        if type(outcome) != ndarray:
            raise ValueError("outcome must be of type numpy.ndarray.")

        # store the outcome
        self._raw_outcome = outcome

        # get the number of qubits
        self._num_qubits = self._raw_outcome.shape[1]

        # get the number of shots
        self._shots = self._raw_outcome.shape[0]

    @property
    def raw_outcome(self):
        """Returns the initial, raw outcome used to instantiate the class."""
        return self._raw_outcome

    @property
    def num_qubits(self):
        """Returns the number of qubits in the measurement outcome."""
        return self._num_qubits

    @property
    def shots(self):
        """Returns the number of shots -- i.e., the number of times
        the circuit was simulated to obtain measurement results.
        """
        return self._shots

    def as_int(self, index):
        """Returns the integer value of a bit string.

        Args:
            index : int
                Index of the bit string to convert to an integer.

        """
        # TODO: there must be a better way of writing this!
        # get the bit string
        bit_string = self[index]

        # string to store the bits
        string = ""

        # add the bits to the string
        for b in bit_string:
            if b == 0:
                string += "0"
            else:
                string += "1"

        return int(string, 2)

    # TODO: implement
    def average_outcome(self):
        """Returns the average over all sampled bit strings."""
        pass

    def __getitem__(self, index):
        """Returns the sampled outcome for the given index."""
        # input checking
        if type(index) != int:
            try:
                index = int(index)
            except (ValueError, TypeError) as error:
                raise error

        # make sure the index is valid
        if index < 0 or index > self._shots:
            raise ValueError("Index out of range.")

        # return the desired raw measurement outcome
        return self._raw_outcome[index, :]

    def __len__(self):
        """Returns the number of shots/samples in a measurment outcome."""
        return self.shots
