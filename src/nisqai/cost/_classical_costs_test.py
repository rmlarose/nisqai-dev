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

# TODO: test classes/methods/functions in _classical_costs.py!

from ._classical_costs import Metrics, DistributionCostFunctions, indicator
import numpy as np
import unittest
class TestSingleInputClassicalCosts(unittest.TestCase):
    '''Testing for classical costs with only scalar valued inputs'''

    def test_indicator_function(self):
        """Test to check if indicator function works properly"""

        value = indicator(0, 1)
        self.assertEqual(value, 1)

class TestClassicalProbabilityCosts(unittest.TestCase):
    '''Testing for Metric/DistributionCostFunctions class of classical costs
        where inputs are probability distribution vectors.
    '''

    def test_l1_distance(self):
        """Test if l1 distance is computed correctly"""

        network_distribution = np.array([0, 1, 24])
        known_distribution = np.array([3, -5, 12])
        l1_correct = 21
        metric_1 = Metrics(network_distribution, known_distribution)
        self.assertAlmostEqual(metric_1.l1_distance(), l1_correct)

        self.assertAlmostEqual(metric_1.lp_distance(1), l1_correct)

        dist_1 = DistributionCostFunctions(network_distribution, known_distribution)

        self.assertAlmostEqual(dist_1.lp_distance(1), l1_correct)

    def test_l1_distance_dict(self):
        """Test if l1 distance is computed correctly for a dict input"""

        network_distribution = {'00': 0, '01': -17,'11': 24}
        known_distribution = {'00': 3, '01': 5,'11': 12}
        l1_correct_dict = 37
        metric_1_dict = Metrics(network_distribution, known_distribution)
        self.assertAlmostEqual(metric_1_dict.l1_distance(), l1_correct_dict)

        self.assertAlmostEqual(metric_1_dict.lp_distance(1), l1_correct_dict)

        dist_1_dict = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_1_dict.lp_distance(1), l1_correct_dict)

    def test_l2_distance(self):
        """Test if l2 distance is computed correctly"""

        network_distribution = np.array([0, 1, 24])
        known_distribution = np.array([3, -5, 12])
        l2_correct = 3*np.sqrt(21)
        metric_2 = Metrics(network_distribution, known_distribution)
        self.assertAlmostEqual(metric_2.l2_distance(), l2_correct)

        self.assertAlmostEqual(metric_2.lp_distance(2), l2_correct)

        dist_2 = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_2.lp_distance(2), l2_correct)

    def test_l2_distance_dict(self):
        """Test if l2 distance is computed correctly for a dict input"""

        network_distribution = {'00': 0, '01': -17,'11': 24}
        known_distribution = {'00': 3, '01': 5,'11': 12}
        l2_correct_dict = 7*np.sqrt(13)
        metric_2_dict = Metrics(network_distribution, known_distribution)
        self.assertAlmostEqual(metric_2_dict.l2_distance(), l2_correct_dict)

        self.assertAlmostEqual(metric_2_dict.lp_distance(2), l2_correct_dict)

        dist_2_dict = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_2_dict.lp_distance(2), l2_correct_dict)

    def test_lp_distance(self):
        """Test if l1 distance is computed correctly"""

        network_distribution = np.array([12, 0.4, -1])
        known_distribution = np.array([42, -18, 7])
        lp_correct = 30.511130402087996
        order = 5
        metric_p = Metrics(network_distribution, known_distribution)
        self.assertAlmostEqual(metric_p.lp_distance(order), lp_correct)

        dist_p = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_p.lp_distance(5), lp_correct)

    def test_lp_distance_dict(self):
        """Test if l1 distance is computed correctly for a dict input"""

        network_distribution = {'00': 9, '01': -3,'11': 71}
        known_distribution = {'00': 24, '01': -1,'11': 2.34}
        lp_correct_dict = 68.6668328668146
        order = 5
        metric_p_dict = Metrics(network_distribution, known_distribution)
        self.assertAlmostEqual(metric_p_dict.lp_distance(order), lp_correct_dict)

        dist_p_dict = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_p_dict.lp_distance(5), lp_correct_dict)

    def test_linf_distance(self):
        """Test if l1 distance is computed correctly"""

        network_distribution = np.array([12, 0.4, -1])
        known_distribution = np.array([42, -18, 7])
        linf_correct = 30
        metric_inf = Metrics(network_distribution, known_distribution)
        self.assertAlmostEqual(metric_inf.linf_distance(), linf_correct)

        dist_inf = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_inf.linf_distance(), linf_correct)
        
        self.assertAlmostEqual(dist_inf.lp_distance("inf"), linf_correct)

    def test_linf_distance_dict(self):
        """Test if l1 distance is computed correctly"""
        
        network_distribution = {'00': 9, '01': -3,'11': 71}
        known_distribution = {'00': 24, '01': -1,'11': 2.34}
        linf_correct_dict = 68.66
        metric_inf_dict = Metrics(network_distribution, known_distribution)
        self.assertAlmostEqual(metric_inf_dict.linf_distance(), linf_correct_dict)

        dist_inf_dict = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_inf_dict.linf_distance(), linf_correct_dict)


    def test_cross_entropy(self):
        """Test if cross_entropy (of network_distribution
            relative to known_distribution) is computed correctly"""

        network_distribution = np.array([0.4, 0.1, 0.5])
        known_distribution = np.array([0.3, 0.27, 0.43])
        x_ent_correct = 1.4924788239639217
        order = 5

        dist_x_ent = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_x_ent.cross_entropy(), x_ent_correct)

    def test_cross_entropy_dict(self):
        """Test if cross_entropy  (of network_distribution
            relative to known_distribution) is computed correctly for a dict input"""

        network_distribution = {'00': 0.2, '01': 0.5,'11': 0.3}
        known_distribution = {'00': 0.1, '01': 0.83,'11': 0.07}
        x_ent_correct_dict = 1.9497443785065087

        dist_x_ent = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_x_ent.cross_entropy(), x_ent_correct_dict)

    def test_cross_entropy_reverse(self):
        """Test if cross_entropy (of network_distribution
            relative to known_distribution) is computed correctly"""

        network_distribution = np.array([0.4, 0.1, 0.5])
        known_distribution = np.array([0.4, 0.49, 0.11])
        x_ent_reverse_correct = 2.2665160044497523

        dist_x_ent_dict = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_x_ent_dict.cross_entropy_reverse(), x_ent_reverse_correct)

    def test_cross_entropy_reverse_dict(self):
        """Test if cross_entropy_reverse (of known distribution
            relative to network_distribution) is computed correctly for a dict input"""

        network_distribution = {'00': 0.2, '01': 0.5,'11': 0.3}
        known_distribution = {'00': 0.1, '01': 0.83,'11': 0.07}
        x_ent_reverse_correct_dict = 1.1837804010803705

        dist_x_ent_rev_dict = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_x_ent_rev_dict.cross_entropy_reverse(), x_ent_reverse_correct_dict)

    def test_kl_divergence(self):
        """Test if kl_divergence (of network_distribution
            relative to known_distribution) is computed correctly"""

        network_distribution = np.array([0.4, 0.1, 0.5])
        known_distribution = np.array([0.3, 0.27, 0.43])
        kl_div_correct = 0.13151477652024035
        order = 5

        dist_kl_div = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_kl_div.kl_divergence(), kl_div_correct)

    def test_kl_divergence_dict(self):
        """Test if kl_divergence (of network_distribution
            relative to known_distribution) is computed correctly for a dict input"""

        network_distribution = {'00': 0.2, '01': 0.5,'11': 0.3}
        known_distribution = {'00': 0.1, '01': 0.83,'11': 0.07}
        kl_div_correct_dict = 0.46426908127917427

        dist_kl_div_dict = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_kl_div_dict.kl_divergence(), kl_div_correct_dict)

    def test_kl_divergence_reverse(self):
        """Test if kl_divergence_reverse (of network_distribution
            relative to known_distribution) is computed correctly"""

        network_distribution = np.array([0.4, 0.1, 0.5])
        known_distribution = np.array([0.4, 0.49, 0.11])
        kl_div_reverse_correct = 0.8831763542965274

        dist_kl_div_rev = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_kl_div_rev.kl_divergence_reverse(), kl_div_reverse_correct)

    def test_kl_divergence_reverse_dict(self):
        """Test if kl_divergence_reverse (of known distribution
            relative to network_distribution) is computed correctly for a dict input"""

        network_distribution = {'00': 0.2, '01': 0.5,'11': 0.3}
        known_distribution = {'00': 0.1, '01': 0.83,'11': 0.07}
        kl_div_reverse_correct_dict = 0.3599145933563619

        dist_kl_div_rev_dict = DistributionCostFunctions(network_distribution, known_distribution)
        
        self.assertAlmostEqual(dist_kl_div_rev_dict.kl_divergence_reverse(), kl_div_reverse_correct_dict)

    def test_linf_dist_value_error(self):
        """Check if string other than 'inf' inputted to lp_distance"""
        network_distribution = np.array([0.4, 0.1, 0.5])
        known_distribution = np.array([0.4, 0.49, 0.11])
        dist = DistributionCostFunctions(network_distribution, known_distribution)

        with self.assertRaises(ValueError):
            dist.lp_distance('not_inf_string')

    def test_add_keys(self): 
        """Test to check if missing keys are added properly"""
        network_distribution = {'00': 0.2, '01': 0.5,'12': 0.3}
        known_distribution = {'00': 0.1, '11': 0.83,'11': 0.07}

        dist = DistributionCostFunctions(network_distribution, known_distribution)
        metric = Metrics(network_distribution, known_distribution)

        self.assertEqual(dist.known_distribution.keys(), dist.network_distribution.keys())

    def test_incorrect_input_types(self):
        network_distribution = {'00': 0.2, '01': 0.5,'12': 0.3}
        known_distribution = np.array([0.4, 0.49, 0.11])

        with self.assertRaises(TypeError):
            dist = DistributionCostFunctions(network_distribution, known_distribution)
        with self.assertRaises(TypeError):
            metric = Metrics(network_distribution, known_distribution)
    
        # Check if lists are inputted
        network_distribution = [0.2, 0.5, 0.3]
        known_distribution = [0.4, 0.49, 0.11]

        with self.assertRaises(TypeError):
            dist = DistributionCostFunctions(network_distribution, known_distribution)

        with self.assertRaises(TypeError):
            metric = Metrics(network_distribution, known_distribution)
    

    def test_numpy_array_sizes(self):
        """Test to check if input numpy arrays have the same size, and are vectors"""

        network_distribution_one = np.array([[0.4, 0.49, 0.11], [3,2,3]])
        known_distribution_one = np.array([0.4, 0.49, 0.11])

        with self.assertRaises(ValueError):
            Metrics(network_distribution_one, known_distribution_one)

        network_distribution_one = np.array([[0.4, 0.49, 0.11], [3,2,3]])
        known_distribution_two = np.array([[0.4, 0.49, 0.11], [3,2,3]])

        with self.assertRaises(ValueError):
            Metrics(network_distribution_one, known_distribution_two)

    def test_incorrect_lp_order(self):

        network_distribution = np.array([0.52, 0.17, 0.31])
        known_distribution = np.array([0.4, 0.49, 0.11])
        mets =  Metrics(network_distribution, known_distribution)

        with self.assertRaises(ValueError):
            mets.lp_distance(0)

        float_order = 3.2441
        with self.assertRaises(TypeError):
            mets.lp_distance(float_order)

    def test_costs_not_defined(self):
        """Tests Kl/x_entropy undefined if both distributions not both zero on a key"""
        network_distribution_one = {'00': 0.2, '01': 0.81,'11': 0.3}
        known_distribution_one = {'00': 0.1, '01': 0,'11': 0.07}
        dist =  DistributionCostFunctions(network_distribution_one, known_distribution_one)

        with self.assertRaises(ValueError):
            dist.kl_divergence()

        network_distribution_two = {'00': 0.2, '01': 0,'11': 0.3}
        known_distribution_two = {'00': 0.1, '01': 0.83,'11': 0.07}
        dist_two =  DistributionCostFunctions(network_distribution_two, known_distribution_two)

        with self.assertRaises(ValueError):
            dist_two.kl_divergence_reverse()


if __name__ == "__main__":
    unittest.main()

