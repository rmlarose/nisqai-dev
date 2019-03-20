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

from pyquil.quil import percolate_declares
import pyquil


def order(program):
    """Orders Quil instructions into a nominal form."""
    # TODO: define nominal form and add more ordering conditions
    # TODO: right now, this just means all DECLARE statements are at the top
    return percolate_declares(program)


def ascii_drawer_simple(program, padlen=2, rebind={'MEASURE': 'MSR'}):
    """Creates an ascii circuit from a pyquil program.

    Idea is to store a single line (a wire) for each qubit. Then, we the lists of lists
    into a string that has gates in order listed in program.

    Inputs
    ------------------------------------------
    program: pyquil program
    padlen: determines number of - padded in between gates
    rebind: can bebind pyquil default names to your own

    Outputs
    -------------------------------------------
    strcirc - an ascii representation of circuit

    TODO: Fix full list of asym for all 2-qubit circuits with asymmetry
    TODO: Fix rebinds to work for gates with parametric dependence (i.e. RX(50)).
    TODO: maybe? Add lines below qubit lines to add connections between two qubit gates?
    TODO: maybe? Is relying on \t being 4 spaces to ignore DECALRE matrix okay?
    """
    # keep track of 'directives'
    directives = ['PRAGMA', 'MEASURE', 'DEFGATE', 'DECLARE']

    # init padding and circuit
    pad = ''.join('-' for n in range(padlen))
    circ = []

    # conventions for asymmetric 2-qubit gates (i.e. CNOT) always placed on FIRST
    # argument; otherwise, padded with space
    asym = {'CNOT': "'"}
    for key in rebind.keys():
        if key in asym:
            asym[rebind[key]] = asym[key]

    # prepares the ascii string by adding qubits and padding with lines
    qubits = program.get_qubits()
    for qubit in qubits:
        circ.append([str(qubit) + ' |0> '])

    # breaks the program up by newline
    steps = program.out().split("\n")

    # iterates over the steps and creates the "ASCII string"
    for step in steps:
        # redefine step by splitting it up
        step = step.split(' ')
        # handle directives
        if step[0] in directives:
            if step[0] == 'PRAGMA':
                continue
            elif step[0] == 'DECLARE':
                continue
            elif step[0] == 'DEFGATE':
                continue
            elif step[0] == 'MEASURE':
                meas, q, reg = step
                if meas in rebind:
                    meas = rebind[meas]
                for count, qubit in enumerate(qubits):
                    if int(q) == qubit:
                        circ[count][0] += (pad + meas + pad)
                    else:
                        midpad = ''.join('-' for n in range(len(meas)))
                        circ[count][0] += (pad + midpad + pad)
        # handle gates
        else:
            # check if 2 qubit gate and make appends to circ
            if len(step) == 3:
                gate, qi, qj = step
                if gate in rebind:
                    gate = rebind[gate]
                for count, qubit in enumerate(qubits):
                    if int(qi) == qubit:
                        agate = gate
                        if gate in asym:
                            agate += asym[gate]
                        circ[count][0] += (pad + agate + pad)

                    elif int(qj) == qubit:
                        agate = gate
                        if gate in asym:
                            agate += ' '
                        circ[count][0] += (pad + agate + pad)

                    else:
                        midpad = ''.join('-' for n in range(len(gate)))
                        if gate in asym:
                            midpad += '-'
                        circ[count][0] += (pad + midpad + pad)

            # check if 1 qubit gate and make appends to circ
            if len(step) == 2:
                gate, q = step
                if gate in rebind:
                    gate = rebind[gate]
                for count, qubit in enumerate(qubits):
                    if int(q) == qubit:
                        circ[count][0] += (pad + gate + pad)
                    else:
                        midpad = ''.join('-' for n in range(len(gate)))
                        circ[count][0] += (pad + midpad + pad)

    # do post-processing on circ lists to make a string for printing purposes
    strcirc = ''
    for line in circ:
        strcirc += line[0]
        strcirc += '\n'

    return strcirc


def ascii_drawer(program, padlen=4):
    """Creates an ascii circuit from a pyquil program.

    Idea is to store a single line (a wire) for each qubit as a diciontary to a string.
    Then a final string is contructed with \n between qubits for printing.

    Inputs
    ------------------------------------------
    program: pyquil program
    padlen: determines number of - padded in between gates

    Outputs
    -------------------------------------------
    strcirc - an ascii representation of circuit

    TODO: add asym?
    TODO: add rebinds to bring down pad length from left (resulting from longest gate)
    """
    gateorder = {qubit: [] for qubit in program.get_qubits()}
    gatenum = {qubit: [-1] for qubit in program.get_qubits()}
    longest_gate_len = 0
    for gate in program:
        gate_name_len = len(gate.out().split(' ')[0])
        if gate_name_len > longest_gate_len:
            longest_gate_len = gate_name_len
        if isinstance(gate, pyquil.quilbase.Gate):
            qubits = gate.qubits
            # pre-processing for 2 qubit gates
            if len(qubits) == 2:
                # find the higher gatenum between the qubit's wire-paths
                first_num = gatenum[int(qubits[0].out())][-1]
                second_num = gatenum[int(qubits[0].out())][-1]
                if first_num > second_num:
                    higher_num = first_num
                else:
                    higher_num = second_num
                # assign this higher number + 1 to BOTH
                gatenum[int(qubits[0].out())].append(higher_num + 1)
                gatenum[int(qubits[1].out())].append(higher_num + 1)
                # add the gate to each qubit's wire
                gateorder[int(qubits[0].out())].append(gate.out().split(' ')[0])
                gateorder[int(qubits[1].out())].append(gate.out().split(' ')[0])
            # pre-processing for 1 qubit gates
            else:
                # increment the gate number for this qubit's wire
                prev_num = gatenum[int(qubits[0].out())][-1]
                gatenum[int(qubits[0].out())].append(prev_num + 1)
                # add the gate to the qubits's wire
                gateorder[int(qubits[0].out())].append(gate.out().split(' ')[0])

    most_gates_key = max(gatenum, key=lambda x: len(set(gatenum[x])))
    rpad = ''.join('-' for n in range(padlen))
    # conventions for asymmetric 2-qubit gates (i.e. CNOT) always placed on FIRST
    # argument; otherwise, padded with space
    qubits = program.get_qubits()
    circ = {qubit: '' for qubit in qubits}

    for num in gatenum[most_gates_key]:
        for qubit in qubits:
            if num == -1:
                circ[qubit] += ('|' + str(qubit) + '>-')
            else:
                if num in gatenum[qubit]:
                    gate = gateorder[qubit].pop(0)
                    gate_len = len(gate)
                    diff = longest_gate_len - gate_len
                    lpad = ''.join('-' for n in range(diff))
                    circ[qubit] += (lpad + gate + rpad)
                else:
                    lpad = ''.join('-' for i in range(longest_gate_len))
                    circ[qubit] += (lpad + rpad)

    # do post-processing on circ lists to make a string for printing purposes
    strcirc = ''
    for key in circ:
        strcirc += circ[key]
        strcirc += '\n'

    return strcirc
