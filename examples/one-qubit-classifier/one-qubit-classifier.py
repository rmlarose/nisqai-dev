"""Implements the Single Qubit Perceptron using the NISQAI library."""

# =======
# imports
# =======

import nisqai

# ========
# get data
# ========

# returns a CData object with 2 features, five samples, and labels
cdata = nisqai.data.random_data(4, 5, [1, 0, 0, 1, 0])

# ===============
# encode the data
# ===============

# get a feature map
feature_map = nisqai.nearest_neighbor(4, 2)

# get an encoder
encoder = nisqai.angle_simple_linear

# encode the data using the AngleEncoding
state_prep = nisqai.AngleEncoding(cdata, encoder, feature_map)

# write the circuit
state_prep._write_circuit(0)

print(state_prep)
