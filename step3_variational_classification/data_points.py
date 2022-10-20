"""
	Generates 20 training data points,
	5 test data points with 2 features.
"""


from qiskit.utils import algorithm_globals
import numpy as np
from qiskit_machine_learning.datasets import ad_hoc_data


algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)

TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = (
	ad_hoc_data(training_size=10,
		test_size=5, n=2, gap=0.3, one_hot=False
		))
