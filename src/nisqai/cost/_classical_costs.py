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

# TODO: write classical cost methods based on the output of the circuit and the data
# for example, indicator cost function based on measurement results as in the one-qubit-classifier

# should this class take in a "network" or "trainer" class?

import numpy as np
import time
from numpy import linalg as LA

def indicator(prediction, label):
    """Returns one if the prediction doesn't match the label, else zero.

    Args:

        prediction : int
            Prediction of a data point by a classifier.

        label : int
            Actual label of a data point by a classifier.
    """
    return 1 if prediction != label else 0


class Metrics:
    """Class of metrics/distances (norms) between probability distributions"""

    def __init__(self, network_distribution, known_distribution):
        """Initializes a metric to compare distance between network and known (data) distribution.
        
        Args:
            network_distribution : dict or np.ndarray
                Output probability distribution from network circuit.

            known_distribution : dict or np.ndarray
                Actual probability distribution from data.
        """
        if type(network_distribution) is not type(known_distribution):
            raise TypeError("Input distributions must be of same type")

        if isinstance(network_distribution, dict):
            if network_distribution.keys() != known_distribution.keys():
                # If network_distribution has a key not in known_distribution 
                # Assign zero to that key value
                for key in list(network_distribution.keys()):   
                    if key not in list(known_distribution.keys()):
                        known_distribution[key] = 0

                # Check keys again and repeat the other way around
                if network_distribution.keys() != known_distribution.keys():
                    for key in list(known_distribution.keys()):   
                        if key not in list(network_distribution.keys()):
                            network_distribution[key] = 0

        elif isinstance(network_distribution, np.ndarray):
            if network_distribution.shape != known_distribution.shape:
                raise ValueError("The probability vectors have different sizes")
            if network_distribution.ndim != 1 or known_distribution.ndim != 1:
                raise ValueError("The arrays must be 1 dimensional.")

        else: raise TypeError("The input distributions must be either a dict or a numpy array")

        self.network_distribution = network_distribution
        self.known_distribution = known_distribution

    def l1_distance(self):
        """Returns l1 distance between probability vectors.
                l2 = Σ_i|x_i -y_i|
        """
        
        if isinstance(self.network_distribution, dict): 
            l1 = 0
            for key in list(self.known_distribution.keys()):

                l1 += abs(self.network_distribution[key] - self.known_distribution[key])

        elif  isinstance(self.network_distribution, np.ndarray):
            l1 = LA.norm(self.network_distribution - self.known_distribution, 1)

        return l1
    
    def l2_distance(self):
        """Returns l2 distance between probability vectors:
            l2 = (Σ_i|x_i -y_i|^2)^{1/2}
        """

        if isinstance(self.network_distribution, dict): 
            l2_abs_summand = 0
            for key in list(self.known_distribution.keys()):
                l2_abs_summand += abs( self.network_distribution[key]
                                        - self.known_distribution[key] )**2

                l2 = l2_abs_summand**(1/2)

        elif  isinstance(self.network_distribution, np.ndarray):
            
            l2 = LA.norm(self.network_distribution - self.known_distribution, 2)

        return l2


    def lp_distance(self, p):
        """Returns lp distance between probability vectors.
                lp= (Σ_i|x_i -y_i|^p)^{1/p}
        """

        if not isinstance(p, int):
            raise TypeError("The order of the distance must be an integer")
        if p == 0:
            raise ValueError("Order value = 0 is not supported")
        if  p == 1:
            lp_distance = self.l1_distance()
        elif p == 2:
            lp_distance = self.l2_distance()
        else:

            if isinstance(self.network_distribution, dict): 
                lp_distance_summand = 0
                for key in list(self.known_distribution.keys()):

                    lp_distance_summand += abs(self.network_distribution[key]
                                                - self.known_distribution[key])**p

                lp_distance = lp_distance_summand**(1/p)

            elif  isinstance(self.network_distribution, np.ndarray):
                lp_distance = LA.norm(self.network_distribution - self.known_distribution, p)

        return lp_distance

    def linf_distance(self):
        """Returns l_inf distance between probability vectors
            l_inf = max(|x_1-y_1|, |x_2-y_2|, ... , |x_n - y_n|)
        """

        if isinstance(self.network_distribution, dict): 
            linf_distance_summand_dict = {}
            for key in list(self.known_distribution.keys()):

                linf_distance_summand_dict[key] = abs(self.network_distribution[key]   \
                                                        - self.known_distribution[key])

            l_inf = max(linf_distance_summand_dict.values())

        elif  isinstance(self.network_distribution, np.ndarray):
            l_inf = LA.norm(self.network_distribution - self.known_distribution, np.inf)

        return l_inf


class DistributionCostFunctions(Metrics):
    """Class of cost functions between probability distributions"""

        
    def __init__(self, network_distribution, known_distribution):
        """Initializes a cost function (which does not have to be a distance)
            to compare distance between network and known (data) distribution.
        
        Args:
            network_distribution : dict or np.ndarray
                Output probability distribution from network circuit.

            known_distribution : dict or np.ndarray
                Actual probability distribution from data.
        """
        super().__init__(network_distribution, known_distribution)


    def cross_entropy(self):
        """ Returns cross entropy between probability vectors
            Convention will be H(network_distribution, known_distribution)  
                = -Σ network_distribution*log(known_distribution)

            For the reverse: H(known_distribution, network_distribution)
            see cross_entropy_reverse      
        """

        if isinstance(self.network_distribution, dict): 
            x_entropy = 0
            for key in list(self.known_distribution.keys()):

                    x_entropy -= self.network_distribution[key]             \
                                    * np.log2(self.known_distribution[key])

        elif  isinstance(self.network_distribution, np.ndarray):
            x_entropy = -np.sum(np.multiply(self.network_distribution,          \
                                                np.log2(self.known_distribution) ) )

        return x_entropy

    def cross_entropy_reverse(self):
        """ Returns cross entropy between probability vectors
            Convention will be H(known_distribution, network_distribution)  
                = -Σ known_distribution*log(network_distribution)

            For the reverse: H(network_distribution, known_distribution)
            see cross_entropy      
        """

        if isinstance(self.network_distribution, dict): 
            x_entropy_rev = 0
            for key in list(self.known_distribution.keys()):

                    x_entropy_rev -= self.known_distribution[key]   \
                                        * np.log2(self.network_distribution[key])
    
        elif  isinstance(self.network_distribution, np.ndarray):
            x_entropy_rev = -np.sum(np.multiply(self.known_distribution,               \
                                                     np.log2(self.network_distribution) ) )

        return x_entropy_rev


    def kl_divergence(self):
        """ Returns KL Divergence between probability vectors
            Convention will be KL(network_distribution || known_distribution)  
                = Σ network_distribution*log(network_distribution/known_distribution)

            For the reverse: KL(known_distribution || network_distribution)
            see kl_divergence_reverse     
        """

        if isinstance(self.network_distribution, dict): 

            kl_div = 0
            for key in list(self.known_distribution.keys()):

                if self.known_distribution[key] == 0:
                    if self.network_distribution[key] != 0:
                        raise ValueError("The KL Divergence is not defined for these values")
                else:
                    kl_div += self.network_distribution[key]                \
                                * np.log2(self.network_distribution[key]    \
                                / self.known_distribution[key])

        elif  isinstance(self.network_distribution, np.ndarray):

            kl_div = np.sum(np.multiply(self.network_distribution,      \
                                np.log2(self.network_distribution           \
                                    / self.known_distribution) ) )

        return kl_div

    def kl_divergence_reverse(self):
        """ Returns KL Divergence between probability vectors
            Convention will be KL(network_distribution || known_distribution)  
                = Σ network_distribution*log(network_distribution/known_distribution)

            For the reverse: KL(known_distribution || network_distribution)
            see kl_divergence_reverse     
        """

        if isinstance(self.network_distribution, dict): 

            kl_div_rev = 0
            for key in list(self.known_distribution.keys()):
                    if self.network_distribution[key] == 0:
                        if self.known_distribution[key] != 0:
                            raise ValueError('The KL Divergence is not defined for these values')
 
                    else:
                        kl_div_rev += self.known_distribution[key]              \
                                        * np.log2( self.known_distribution[key] \
                                        / self.network_distribution[key] )
      
        elif  isinstance(self.network_distribution, np.ndarray):

            kl_div_rev = np.sum(np.multiply(self.known_distribution,            \
                                                np.log2(self.known_distribution     \
                                                    / self.network_distribution) ) )

        return kl_div_rev

    def lp_distance(self, order):
        """"Returns a probability distance between network and known (data) distribution."""

        if isinstance(order, str):
            if not order.lower() == "inf":
                raise ValueError("Only the infinity norm is supported")
            else:
                distance = Metrics.linf_distance(self)

        elif isinstance(order, int) and order != 0:
            distance = Metrics.lp_distance(self, order)

        return distance