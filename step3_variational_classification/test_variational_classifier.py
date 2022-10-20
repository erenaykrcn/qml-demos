from variational_classification import get_classification_probabilities, opt_theta
from data_points import TEST_DATA, TEST_LABELS, TRAIN_LABELS, TRAIN_DATA


def classify_test_data(x_data, y_data, theta):
	probs = get_classification_probabilities(x_data, theta)
	predictions = [0 if p[0] >= p[1] else 1 for p in probs]

	accuracy = 0
	for i, prediction in enumerate(predictions):
		if prediction == y_data[i]:
			accuracy += 1
	accuracy /= len(y_data)

	return accuracy, predictions

accuracy, predictions = classify_test_data(TEST_DATA, TEST_LABELS, opt_theta)


"""
	Visualizing the training data and test data.
"""

from matplotlib.lines import Line2D
plt.figure(figsize=(9, 6))

for feature, label in zip(TRAIN_DATA, TRAIN_LABELS):
	COLOR = 'C0' if label == 0 else 'C1'
	plt.scatter(feature[0], feature[1],
		marker='o', s=100, color=COLOR)

for feature, label, pred in zip(TEST_DATA, TEST_LABELS, predictions):
	COLOR = 'C0' if pred == 0 else 'C1'
	plt.scatter(feature[0], feature[1],
			marker='s', s=100, color=COLOR)
	if label != pred:  # mark wrongly classified
		plt.scatter(feature[0], feature[1], marker='o', s=500,
			linewidths=2.5, facecolor='none', edgecolor='C3')

legend_elements = [
	Line2D([0], [0], marker='o', c='w', mfc='C0', label='A', ms=10),
	Line2D([0], [0], marker='o', c='w', mfc='C1', label='B', ms=10),  # MISTAKE!
	Line2D([0], [0], marker='s', c='w', mfc='C1', label='predict A',
			ms=10),
	Line2D([0], [0], marker='s', c='w', mfc='C0', label='predict B',
			ms=10),
	Line2D([0], [0], marker='o', c='w', mfc='none', mec='C3',
		label='wrongly classified', mew=2, ms=15)
]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')

plt.title('Training & Test Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.savefig("training_and_test_data.png")
