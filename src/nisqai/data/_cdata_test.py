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
from numpy import array, array_equal, allclose

from nisqai.data._cdata import CData, LabeledCData, random_data, get_iris_setosa_data, get_mnist_data

import unittest


class DataTest(unittest.TestCase):
    """Unit tests for CData and LabeledCData."""
    def test_basic_cdata(self):
        """Creates a CData object and makes sure the dimensions are correct."""
        data = array([[1, 0, 0], [0, 1, 0]])
        cdata = CData(data)
        self.assertEqual(cdata.num_features, 3)
        self.assertEqual(cdata.num_samples, 2)

    # TODO: This is not supported -- why do we have this?
    # def test_higher_dim_cdata(self):
    #     """Creates a (3x3x3) Cdata object and
    #     ensures dimensions are correct."""
    #     data = array([[[0, 1, 2],
    #                    [3, 4, 5],
    #                    [6, 7, 8]],
    #                   [[9, 10, 11],
    #                    [12, 13, 14],
    #                    [15, 16, 17]],
    #                   [[18, 19, 20],
    #                    [21, 22, 23],
    #                    [24, 25, 26]]])
    #     cdata = CData(data)
    #     self.assertEqual(cdata.num_samples, 3)
    #     self.assertEqual(cdata.num_features, 9)

    def test_basic_labeled_cdata(self):
        """Creates a LabeledCData object and makes sure the dimensions are correct."""
        data = array([[1, 0, 0], [0, 1, 0]])
        labels = array([1, 0])
        lcdata = LabeledCData(data, labels)
        self.assertEqual(lcdata.num_features, 3)
        self.assertEqual(lcdata.num_samples, 2)

    def test_get_random_data_basic(self):
        """Tests to see if we can get random data."""
        cdata = random_data(num_features=2,
                            num_samples=4,
                            labels=None)
        self.assertEqual(cdata.num_features, 2)
        self.assertEqual(cdata.num_samples, 4)

    def test_scale_features_min_max_norm(self):
        """Tests min-max norm method of scale_features."""
        data = array([[0.564, 20.661], [-18.512, 41.168], [-0.009, 20.440]])
        cdata = CData(data)

        # correct answer computed with Mathematica
        # TODO: can we compute the right answer in Python?
        answer = array([[1, 0.0106619], [0, 1], [0.969962, 0]])

        # perform min-max norm scaling on features and check answer
        cdata.scale_features('min-max norm')
        self.assertTrue(allclose(cdata.data, answer))

    def test_scale_features_mean_norm(self):
        """Tests mean norm method of scale_features."""
        data = array([[0.564, 20.661], [-18.512, 41.168], [-0.009, 20.440]])
        cdata = CData(data)

        # correct answer computed in Mathematica
        # TODO: can we compute the right answer in Python?
        answer = array([[0.343346, -0.326225], [-0.656654, 0.663113], [0.313308, -0.336887]])

        # perform mean norm scaling on features and check answer
        cdata.scale_features('mean norm')
        self.assertTrue(allclose(cdata.data, answer))

    def test_scale_features_standardize(self):
        """Tests standardization method of scale_features."""
        data = array([[0.564, 20.661], [-18.512, 41.168], [-0.009, 20.440]])
        cdata = CData(data)

        # correct answer computed in Mathematica
        # TODO: can we compute the right answer in Python?
        answer = array([[0.60355, -0.568043], [-1.1543, 1.15465], [0.550748, -0.586608]])

        # perform standardization feature scaling and check answer
        cdata.scale_features('standardize')
        self.assertTrue(allclose(cdata.data, answer))

    def test_scale_features_L2_norm(self):
        """Tests L2 norm method of scale_features."""
        data = array([[0.564, 20.661], [-18.512, 41.168], [-0.009, 20.440]])
        cdata = CData(data)

        # correct answer computed in Mathematica
        # TODO: can we compute the right answer in Python?
        answer = array([[0.0304526, 0.409996], [-0.999536, 0.816936], [-0.000485946, 0.40561]])

        # perform L2 normalization and check answer
        cdata.scale_features('L2 norm')
        self.assertTrue(allclose(cdata.data, answer))

    def test_scale_features_L1_norm(self):
        """Tests L1 norm method of scale_features."""
        # Get some data
        data = array([[0.564, 20.661], [-18.512, 41.168], [-0.009, 20.440]])
        cdata = CData(data)

        # Correct answer computed in Mathematica
        # TODO: can we compute the right answer in Python?
        answer = array([[0.029552, 0.25114], [-0.969976, 0.500407], [-0.000471575, 0.248453]])

        # Perform L1 normalization and check answer
        cdata.scale_features('L1 norm')
        self.assertTrue(allclose(cdata.data, answer))

    def test_reduce_features_size(self):
        """Ensures PCA gives correct shape."""
        # Get some data
        data = array([[0.564, 20.661, 1], [-18.512, 41.168, -1],
                      [-0.009, 20.440, 7]])
        cdata = CData(data)

        # ===================================
        # Perform PCA to reduce to 2 features
        # ===================================

        # Reduce by nearest int closest to 60%, rounding up
        frac = 0.6
        cdata.reduce_features(frac)
        self.assertTrue(cdata.data.shape == (3, 2))

    def test_keep_labels(self):
        """Tests keeping only a subset of data with certain labels."""
        # Create some arbitrary data and labels
        data = array([[1], [2], [3], [4], [5], [6]])
        labels = array([1, 1, 2, 2, 3, 3])

        # Create a LabeledCData object
        lcdata = LabeledCData(data, labels)

        self.assertTrue(array_equal(lcdata.data, data))
        self.assertTrue(array_equal(lcdata.labels, labels))

        # Make sure 3 is in the labels, for contrast
        self.assertIn(3, lcdata.labels)

        # Only keep the 1 and 2 labels
        lcdata.keep_data_with_labels([1, 2])

        # Make sure 3 has been removed from the labels, for contrast
        self.assertNotIn(3, lcdata.labels)

        # Correct answers
        newdata = array([[1], [2], [3], [4]])
        newlabels = array([1, 1, 2, 2])

        # Make sure the new data is correct
        self.assertTrue(array_equal(lcdata.data, newdata))
        self.assertTrue(array_equal(lcdata.labels, newlabels))

    def test_keep_labels_all(self):
        """Tests keeping only a subset of data with certain labels."""
        # Create some arbitrary data and labels
        data = array([[1], [2], [3], [4], [5], [6]])
        labels = array([1, 1, 2, 2, 1, 2])

        # Create a LabeledCData object
        lcdata = LabeledCData(data, labels)

        self.assertTrue(array_equal(lcdata.data, data))
        self.assertTrue(array_equal(lcdata.labels, labels))

        # Only keep the 1 and 2 labels
        lcdata.keep_data_with_labels([1, 2])

        # Make sure the new data is correct
        self.assertTrue(array_equal(lcdata.data, data))
        self.assertTrue(array_equal(lcdata.labels, labels))

    def test_keep_labels2(self):
        """Tests keeping only a subset of data with certain labels."""
        # Create some arbitrary data and labels
        data = array([[1], [2], [3], [4], [5], [6]])
        labels = array([1, 1, 2, 2, 3, 3])

        # Create a LabeledCData object
        lcdata = LabeledCData(data, labels)

        self.assertTrue(array_equal(lcdata.data, data))
        self.assertTrue(array_equal(lcdata.labels, labels))

        # Make sure 2 is in the labels, for contrast
        self.assertIn(2, lcdata.labels)

        # Only keep the 1 and 3 labels
        lcdata.keep_data_with_labels([1, 3])

        # Make sure 3 has been removed from the labels, for contrast
        self.assertNotIn(2, lcdata.labels)

        # Correct answers
        newdata = array([[1], [2], [5], [6]])
        newlabels = array([1, 1, 3, 3])

        # Make sure the new data is correct
        self.assertTrue(array_equal(lcdata.data, newdata))
        self.assertTrue(array_equal(lcdata.labels, newlabels))

    def test_get_iris_setosa_data(self):
        """Checks that iris setosa data set is of correct length."""
        iris = get_iris_setosa_data()
        self.assertEqual(len(iris.data), 150)
        self.assertEqual(len(iris.labels), 150)

    def test_get_mnist_data(self):
        """Checks that mnist data set is of correct length."""
        # TODO: Remove once get_mnist_data(...) is fixed.
        pass
        # mnist = get_mnist_data()
        # self.assertEqual(len(mnist.data), 60000)
        # self.assertEqual(len(mnist.labels), 60000)

    def test_basic_labeling(self):
        """Ensures LabeledCData runs correctly with given input labels."""
        # data with only 1 feature
        data = array([[-1], [1], [0.5], [0.25], [-0.33], [0]])
        # give 1 if feature value >= 0; otherwise 0
        labels = array([0, 1, 1, 1, 0, 1])
        cdata = LabeledCData(data, labels)

        # ensure that labelling is correct
        assert array_equal(cdata.labels, labels)

    def test_func_labeling(self):
        """Ensure data labelled with a basic function works."""
        # Define a labeling function
        def label(x):
            return 1 if x >= 0 else 0

        # Create (arbitrary) data
        data = array([[500], [-17], [12], [0], [-.002], [.001]])

        # Manually create the labels
        labels = array([label(x) for x in data])

        # Create a labelled cdata object by passing in the labeling function
        cdata = LabeledCData(data, label)

        # Make sure the data is labelled correctly
        self.assertTrue(array_equal(labels, cdata.labels))

    # TODO: The previous input to LabeledCData was not of the correct type.
    #  Hence, the subsequent checks do not make sense when comparing arrays.
    # def test_data_splitting(self):
    #     """Ensures that splitting data into training and test set works."""
    #     # Get data
    #     data = array([[-1, 0],
    #                   [1, 5],
    #                   [-2, 17],
    #                   [1, 2],
    #                   [4, 3]])
    #     labels = array([1, 1, 0, 0, 1])
    #     cdata = LabeledCData(data, labels)
    #
    #     # Split data into various sizes
    #     testdata, traindata = cdata.train_test_split(0.1)
    #     self.assertTrue(array_equal(data[:0], testdata))
    #     self.assertTrue(array_equal(data[1::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(0.2)
    #     self.assertTrue(array_equal(data[:1], testdata))
    #     self.assertTrue(array_equal(data[2::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(0.3)
    #     self.assertTrue(array_equal(data[:2], testdata))
    #     self.assertTrue(array_equal(data[3::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(0.4)
    #     self.assertTrue(array_equal(data[:2], testdata))
    #     self.assertTrue( array_equal(data[3::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(0.5)
    #     self.assertTrue(array_equal(data[:3], testdata))
    #     self.assertTrue( array_equal(data[4::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(0.6)
    #     self.assertTrue(array_equal(data[:4], testdata))
    #     self.assertTrue( array_equal(data[5::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(0.7)
    #     self.assertTrue(array_equal(data[:4], testdata))
    #     self.assertTrue(array_equal(data[5::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(0.8)
    #     self.assertTrue(array_equal(data[:5], testdata))
    #     self.assertTrue( array_equal(data[6::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(0.9)
    #     self.assertTrue(array_equal(data[:6], testdata))
    #     self.assertTrue(array_equal(data[7::], traindata))
    #
    #     testdata, traindata = cdata.train_test_split(1)
    #     self.assertTrue(array_equal(data[:7], testdata))
    #     self.assertTrue(array_equal(data[8::], traindata))

if __name__ == "__main__":
    unittest.main()
