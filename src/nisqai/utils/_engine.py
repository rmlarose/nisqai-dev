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
import socket
import psutil
from pyquil.api import ForestConnection
from subprocess import Popen, DEVNULL, STDOUT, check_output


class DeactivateQUILCError(Exception):
    pass


class DeactivateQVMError(Exception):
    pass


def get_free_port():
    """Returns a free port."""
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# TODO: Would be good to have functions startQVM, startQUILC, stopQVM, stopQUILC separately
def startQVMandQUILC(qvm_executable=None, quilc_executable=None, default_ports=True):
    """Activates qvm and quilc server for running pyQuil programs.

    Args:
        qvm_executable : int
            Path to the qvm server excecutable.

        quilc_executable : str
             Path to the quilc server excecutable.

        default_ports: bool
            Flat to use default ports specified in pyquilConfig.

    Returns:
        qvm_server : Popen object
        quilc_server : Popen object
        forest_connection : ForestConnection object
    """
    # Quantum virtual machine
    if qvm_executable is None:
        qvm_exec = check_output('type qvm', shell=True, universal_newlines=True).split(
            '\n')[0].split(' ')[-1]
    else:
        qvm_exec = qvm_executable

    # Quil compiler
    if quilc_executable is None:
        quilc_exec = check_output(
            'type quilc', shell=True, universal_newlines=True).split('\n')[0].split(' ')[-1]
    else:
        quilc_exec = quilc_executable

    # TODO: Comments here. What's this code doing?
    prev_qvm = check_output(
        'ps aux | grep "' + qvm_exec + ' -S"', shell=True, universal_newlines=True)
    prev_quilc = check_output(
        'ps aux | grep "' + quilc_exec + ' -S"', shell=True, universal_newlines=True)

    # TODO: Comments here. What's this code doing?
    if len(prev_qvm.split('\n')) > 3:
        proc = psutil.Process(int(prev_qvm.split('  ')[1]))
        proc.kill()

    # TODO: Comments here. What's this code doing?
    if len(prev_quilc.split('\n')) > 3:
        proc = psutil.Process(int(prev_quilc.split('  ')[1]))
        proc.kill()

    # TODO: Comments here. What's this code doing?
    if default_ports:
        qvm_server = Popen([qvm_exec, "-S"], stdout=DEVNULL, stderr=STDOUT)
        quilc_server = Popen([quilc_exec, "-S"], stdout=DEVNULL, stderr=STDOUT)
        fc = ForestConnection(sync_endpoint=None,
                              compiler_endpoint=None)
    else:
        # TODO: any issues with using http instead of https?
        local_address = 'http://127.0.0.1:'
        qvm_port = get_free_port()
        quilc_port = get_free_port()
        qvm_server = Popen([qvm_exec, "-S", "-p", str(qvm_port)],
                           stdout=DEVNULL, stderr=STDOUT)
        quilc_server = Popen([quilc_exec, "-S", "-p", str(quilc_port)], 
                           stdout=DEVNULL, stderr=STDOUT)
        fc = ForestConnection(sync_endpoint= local_address + str(qvm_port),
                              compiler_endpoint= local_address + str(quilc_port))
    
    return qvm_server, quilc_server, fc


def stopQVMandQUILC(qvm_server, quilc_server):
    """Deactivates qvm and quilc server.

    Args:
        qvm_server : Popen
            Popen object for the running qvm server.

        quilc_server : Popen
            Popen object for the running quilc server.
    """

    try:
        qvm_server.terminate()

    # TODO: What specifcially are we excepting here?
    except:
        raise Exception("Invalid qvm server.")
    
    try:
        quilc_server.terminate()

    # TODO: What specifcially are we excepting here?
    except:
        raise Exception("Invalid quilc server.")
