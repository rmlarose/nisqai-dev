"""Implements the Single Qubit Perceptron using the NISQAI library."""

# =======
# imports
# =======

import nisqai


# ========
# get data
# ========

# returns a CData object with 2 features, five samples, and labels
cdata = nisqai.data.random_data(2, 5, [1, 0, 0, 1, 0])

# ===============
# encode the data
# ===============


