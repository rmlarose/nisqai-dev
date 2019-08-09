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
import warnings
import psutil
from pyquil.api import ForestConnection
from subprocess import Popen, DEVNULL, STDOUT, check_output


class DeactivateQUILCError(Exception):
    pass

class DeactivateQVMError(Exception):
    pass

class engine:

    # Initializing the servers with default: None
    def __init__(self):
        self.local_address = 'https://127.0.0.1:'
        self.qvm_server = None
        self.qvm_exec = None
        self.qvm_port = None
        self.quilc_server = None
        self.quilc_exec = None
        self.quilc_port = None
        self.forest_connection = None
 
    # Activates the QVM server and stores it in qvm_server
    def startQVM(self, qvm_executable=None, default_port=True):
        """Activates qvm and quilc server for running pyQuil programs.

        Args:
            qvm_executable : str
                Path to the qvm server excecutable.

            default_port: bool
                Flat to use default ports specified in pyquilConfig.

        Returns:

        """
        if self._checkQVM():
            warnings.warn('Skipping... QVM server is already running!')
            # Generates list of all running processes
            proc_list = [x.as_dict(attrs=['pid', 'name'])
                         for x in list(psutil.process_iter())]
            # Finds PID of QVM server process
            pid_data = [x['pid'] for x in proc_list if 'qvm' == x['name']]
            # Stores it into qvm_server
            self.qvm_server = psutil.Process(pid_data[0])

        else:
            # Find the standard path of QVM server else
            # Use the different path which is provided.
            self.qvm_exec = qvm_executable if qvm_executable is not None else check_output(
                'type qvm', shell=True, universal_newlines=True).split('\n')[0].split(' ')[-1]
            
            # Checks whether standard port is to be used
            # Or a new one has to be assigned
            if default_port:
                self.qvm_server = Popen([self.qvm_exec, "-S"], stdout=DEVNULL, stderr=STDOUT)
            else:
                self.qvm_port = self._get_port()
                self.qvm_server = Popen([self.qvm_exec, "-S", "-p", str(self.qvm_port)],
                                   stdout=DEVNULL, stderr=STDOUT)

    # Activates the QUILC server and stores it in quilc_server
    def startQUILC(self, quilc_executable=None, default_port=True):
        """Activates qvm and quilc server for running pyQuil programs.

        Args:
            quilc_executable : str
                Path to the quilc server excecutable.

            default_ports: bool
                Flat to use default ports specified in pyquilConfig.

        Returns:

        """
        if self._checkQUILC():
            warnings.warn('Skipping... QUILC server is already running!')
            # Generates list of all running processes
            proc_list = [x.as_dict(attrs=['pid', 'name'])
                         for x in list(psutil.process_iter())]
            # Finds PID of QUILC server process
            pid_data = [x['pid'] for x in proc_list if 'quilc' == x['name']]
            # Stores it into quilc_server
            self.qvm_server = psutil.Process(pid_data[0])
 
        else:
            # Find the standard path of QUILC server else
            # Use the different path which is provided.
            self.quilc_exec = quilc_executable if quilc_executable is not None else check_output(
                'type quilc', shell=True, universal_newlines=True).split('\n')[0].split(' ')[-1]

            # Checks whether standard port is to be used
            # Or a new one has to be assigned
            if default_port:
                self.quilc_server = Popen([self.quilc_exec, "-R"], stdout=DEVNULL, stderr=STDOUT)
            else:
                self.quilc_port = self._get_port()
                self.quilc_server = Popen([self.quilc_exec, "-R", "-p", str(self.quilc_port)],
                                     stdout=DEVNULL, stderr=STDOUT)
    
    
    def forestObject(self):
        """ Makes an ForestConnection object for running pyQuil programs. """
        if not self._checkQVM():
            raise Warning('Run the QVM server')
        elif not self._checkQUILC():
            raise Warning('Run the QUILC server')
        else:
            qvm_url = self.qvm_port if self.qvm_port is None else self.local_address + \
                str(self.qvm_port)
            quilc_url = self.quilc_port if self.quilc_port is None else self.local_address + \
                str(self.quilc_port)
            # Forest Connections are used when we assign modified ports to the servers
            self.forest_connection = ForestConnection(sync_endpoint = qvm_url,
                                      compiler_endpoint = quilc_url)

    def _get_port(self):
        """ Returns a free port."""
        sock = socket.socket()
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port
    
    def _checkQVM(self):
        """ Checks running status for QVM server
        Args:
        
        Returns: 
            status: bool
                True if any QVM server service is running else return False   
        """
        status = True if self.qvm_server is not None else checkStatusQVM()
        return status

    def _checkQUILC(self):
        """ Checks running status for QVM server
        Args:

        Returns: 
            status: bool
                True if any QVM server service is running else return False   
        """
        status = True if self.quilc_server is not None else checkStatusQUILC()
        return status

    def stopQVM(self):
        """ Stops any running service of QVM server """

        if self._checkQVM():
            self.qvm_server.terminate()
            self.qvm_server.wait()
        else:
            raise Exception('No QVM services running!')
    
    def stopQUILC(self):
        """ Stops any running service of QUILC server """

        if self._checkQUILC():
            self.quilc_server.terminate()
            self.quilc_server.wait()
        else:
            raise Exception('No QUILC services running!')

def checkStatusQVM():
    '''
        Check the status of the QVM server
        Args: 
 
        Return: 
            status : (bool) Status of the qvm server executable
    '''
    # Generates list of all running processes
    proc_list = [x.as_dict(attrs=['pid', 'name'])
                 for x in list(psutil.process_iter())]
    # Checks if any QVM server process exist in the list
    proc_data = ['qvm' == x['name'] for x in proc_list]
    status = True if any(proc_data) else False
    return status

def checkStatusQUILC():
    '''
        Check the status of the QUILC server
        Args: 

        Return: 
            status : (bool) 
                Status of the quilc server executable
    '''
    # Generates list of all running processes
    proc_list = [x.as_dict(attrs=['pid', 'name'])
                 for x in list(psutil.process_iter())]
    # Checks if any QUILC server process exist in the list
    proc_data = ['quilc' == x['name'] for x in proc_list]
    status = True if any(proc_data) else False
    return status
