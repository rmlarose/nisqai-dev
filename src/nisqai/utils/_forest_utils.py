import time
import socket
import psutil
import pyquil
from pyquil.api import ForestConnection
from subprocess import Popen, DEVNULL, STDOUT, check_output

# Get free ports
def get_free_port():
    '''
    Helper function which returns a free port.
    '''
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

# Start compiler and quantum virtual machine
def activate_qvm_quilc(qvm_executable=None, quilc_executable=None, default_ports=True):
    '''
    Activates qvm and quilc server for running pyQuil programs.

    Parameters:
    qvm_executable      (int): Path to the qvm server excecutable.
    quilc_executable    (str): Path to the quilc server excecutable.
    default_ports       (bool): Use default ports specified in pyquilConfig.

    Returns:
    qvm_server:         Popen object
    quilc_server:       Popen object
    fc:                 ForestConnection object

    '''
    if qvm_executable is None:
        qvm_exec = check_output('type qvm', shell=True, universal_newlines=True).split(
            '\n')[0].split(' ')[-1]
    else:
        qvm_exec = qvm_executable

    if quilc_executable is None:
        quilc_exec = check_output(
            'type quilc', shell=True, universal_newlines=True).split('\n')[0].split(' ')[-1]
    else:
        quilc_exec = quilc_executable

    prev_qvm = check_output(
        'ps aux | grep "' + qvm_exec + ' -S"', shell=True, universal_newlines=True)
    prev_quilc = check_output(
        'ps aux | grep "' + quilc_exec + ' -S"', shell=True, universal_newlines=True)
    
    if len(prev_qvm.split('\n')) > 3:
        proc = psutil.Process(int(prev_qvm.split('  ')[1]))
        proc.kill()

    if len(prev_quilc.split('\n')) > 3:
        proc = psutil.Process(int(prev_quilc.split('  ')[1]))
        proc.kill()

    if default_ports:
        qvm_server = Popen([qvm_exec, "-S"], stdout=DEVNULL, stderr=STDOUT)
        quilc_server = Popen([quilc_exec, "-S"], stdout=DEVNULL, stderr=STDOUT)
        fc = ForestConnection(sync_endpoint=None,
                              compiler_endpoint=None)
    else:
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

# Stop compiler and quantum virtual machine
def deactivate_qvm_quilc(qvm_server,quilc_server):
    '''
    Deactivates qvm and quilc server.
    Parameters:
    qvm_server      (Popen): Popen object for the running qvm server.
    quilc_server    (Popen): Popen object for the running quilc server.

    Returns:
    None
    '''

    try:
        qvm_server.terminate()
    except:
        raise Exception('Invalid qvm server.')
    
    try:
        quilc_server.terminate()
    except:
        raise Exception('Invalid quilc server.')
