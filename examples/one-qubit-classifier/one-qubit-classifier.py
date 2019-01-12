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

# get a feature map
feature_map = nisqai.nearest_neighbor(2, 1)

# get an encoder
encoder = nisqai.angle_simple_linear

# encode the data using the AngleEncoding
state_prep = nisqai.AngleEncoding(cdata, encoder, feature_map)

# write the circuit
state_prep._write_circuit(0)

# print out this circuit
print(state_prep.circuits[0])

# =============
# add an ansatz
# =============

ansatz = nisqai.ProductAnsatz(1)

print(ansatz)

# ======================
# add measurement scheme
# ======================


# ===============
# do the training
# ===============
