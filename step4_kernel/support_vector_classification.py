"""
	SVC is a classical ML Algorithm, where we use a feature map to map
	the data points to a higher dimensional space and find the best fitting
	"hyperplane" for seperating the data points into two groups. 

	With the kernel we map form pairs from the data points and calculate the
	transition probabilities to give us a metric on how similar both of the 
	data points are. 
"""
import sys
sys.path.append("../step3_variational_classification")

import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit import opflow
from data_points import TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS


def calculate_kernel(feature_map, x_data, y_data=None):
	"""
		y_data is usually used for testing purpose. With that, the similarity
		of the test data to the train data computed.
	"""

	if y_data == None:
		y_data = x_data

	x_circuits = opflow.CircuitStateFn(feature_map).bind_parameters(
			dict(zip(feature_map.parameters, np.transpose(x_data),tolist()))
		)

	y_circuits = opflow.CircuitStateFn(feature_map).bind_parameters(
			dict(zip(feature_map.parameters, np.transpose(y_data),tolist()))
		)

	kernel = np.abs(
			(~y_circuits.to_matrix_op() @ x_circuits.to_matrix_op()).eval()
		)**2

	return kernel

feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
train_kernel = calculate_kernel(feature_map, TRAIN_DATA)


from sklearn.svm import SVC
# train scikit-learn svm model
model = SVC(kernel='precomputed')
model.fit(train_kernel, TRAIN_LABELS)
print("Number of support vectors for each class:",model.n_support_)
print("Indices of support vectors:", model.support_)

test_kernel = calculate_kernel(feature_map, TRAIN_DATA, TEST_DATA)

# Displays the accuracy of the classification
print(model.score(test_kernel, TEST_LABELS))

