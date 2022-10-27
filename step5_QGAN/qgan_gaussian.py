import numpy as np
import tensorflow as tf
import pickle

from qiskit import QuantumCircuit, Aer
from qiskit.opflow import (StateFn, PauliSumOp, Gradient,
                           AerPauliExpectation)
from qiskit_finance.circuit.library import NormalDistribution
from qiskit.utils import QuantumInstance
from qiskit.circuit import  ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.neural_networks import OpflowQNN
from qiskit.quantum_info import Statevector


# State that we will try to replicate/generate.
data_qubits = 3
real_data = NormalDistribution(data_qubits, mu=0, sigma=0.15)
real_data = real_data.decompose().decompose().decompose()
real_prob_dict = Statevector(real_data).probabilities_dict()


qi_sv = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))

# Generator Ansatz: TwoLocal, full entanglement with two layers
generator = TwoLocal(data_qubits, ['ry','rz'], 'cz', 'full', reps=2, parameter_prefix='θ_g', name="Generator")
THETA_G_DIMS = len(generator.parameters)

# Discriminator Ansatz: cx(qubit, last_qubit)
THETA_D_DIMS = 3*data_qubits+6
discriminator = QuantumCircuit(data_qubits+1, name="Discriminator")
disc_weights = ParameterVector('θ_d', THETA_D_DIMS)

weights = 0

discriminator.rx(disc_weights[weights],data_qubits)
discriminator.rx(disc_weights[weights+1],data_qubits)
discriminator.rx(disc_weights[weights+2],data_qubits)
weights += 3

for qubit in range(data_qubits):
	discriminator.h(qubit)
	discriminator.rx(disc_weights[weights], qubit)
	discriminator.ry(disc_weights[weights + 1], qubit)
	discriminator.rz(disc_weights[weights + 2], qubit)
	discriminator.cx(qubit, data_qubits)
	weights += 3

discriminator.rx(disc_weights[weights],data_qubits)
discriminator.rx(disc_weights[weights+1],data_qubits)
discriminator.rx(disc_weights[weights+2],data_qubits)


# Delivers the Kullback-Leiber Divergence, which is a metric for the 
# similarity of two states. Max similarity corresponds to 0.
def calculate_kl_div(model_distribution: dict, target_distribution: dict):
	"""Gauge model performance using Kullback Leibler Divergence"""
	kl_div = 0
	for bitstring, p_data in target_distribution.items():
		if np.isclose(p_data, 0, atol=1e-8):
			continue
		if bitstring in model_distribution.keys():
			kl_div += (p_data * np.log(p_data)
			 - p_data * np.log(model_distribution[bitstring]))
		else:
			kl_div += p_data * np.log(p_data) - p_data * np.log(1e-6)
	return kl_div


# Gen + Disc Circuit
gen_disc_circuit = QuantumCircuit(data_qubits+1)
gen_disc_circuit.compose(generator, inplace=True)
gen_disc_circuit.compose(discriminator, inplace=True)

# Real Data + Disc Circuit
real_disc_circuit = QuantumCircuit(data_qubits+1)
real_disc_circuit.compose(real_data, inplace=True)
real_disc_circuit.compose(discriminator, inplace=True)

"""
	Instead of using the StateVector function, we get the
	Pauli expectation values of the circuits.
"""

exp_val = AerPauliExpectation()
gradient = Gradient()

gen_disc_sfn = StateFn(gen_disc_circuit)
real_disc_sfn = StateFn(real_disc_circuit)

# Pauli Z operation on the last qubit.
H = StateFn(PauliSumOp.from_list([('ZII', 1.0)]))

gen_disc_op = ~H @ gen_disc_sfn
real_disc_op = ~H @ real_disc_sfn


"""
	Instantiate the Neural Network Models for three scenarios:
	Generated state classified as real with θ_g varying, 
	Generated state classified as real with θ_d varying,
	Real sstate classified as real with θ_d varying.

	Pauli Z Operation on the last qubit has the expectation value 
	between -1 and 1. -1 means discriminator result "real" (|1>)
	and 1 means discriminator result "fake" (|0>). This expectation
	value is tried to be minimized for the trainable parameters.
"""

gen_qnn = OpflowQNN(
		gen_disc_op,
		gen_disc_circuit.parameters[:THETA_D_DIMS],
		gen_disc_circuit.parameters[THETA_D_DIMS:],
		exp_val,
		gradient,
		qi_sv
	)

disc_fake_qnn = OpflowQNN(
		gen_disc_op,
		gen_disc_circuit.parameters[THETA_D_DIMS:],
		gen_disc_circuit.parameters[:THETA_D_DIMS],
		exp_val,
		gradient,
		qi_sv
	)

disc_real_qnn = OpflowQNN(
		real_disc_op,
		[],
		gen_disc_circuit.parameters[:THETA_D_DIMS],
		exp_val,
		gradient,
		qi_sv
	)

"""
	Helper functions to calculate the costs of both models.
	Isnt used in the optimization directly, called for retrieving
	the information for display purposes.
"""

def generator_cost(gen_params):
	# .numpy() method extracts numpy array from TF tensor
	curr_params = np.append(disc_params.numpy(), gen_params.numpy())
	state_probs = Statevector(gen_disc_circuit.bind_parameters(curr_params)
							).probabilities()
	# Get total prob of measuring |1> on q2
	prob_fake_true = np.sum(state_probs[0b100:])
	cost = -prob_fake_true
	return cost

def discriminator_cost(disc_params):
	# .numpy() method extracts numpy array from TF tensor
	curr_params = np.append(disc_params.numpy(), gen_params.numpy())
	gendisc_probs = Statevector(gen_disc_circuit.bind_parameters(curr_params)
							).probabilities()
	realdisc_probs = Statevector(real_disc_circuit.
									bind_parameters(disc_params.numpy())
								).probabilities()
	# Get total prob of measuring |1> on q2
	prob_fake_true = np.sum(gendisc_probs[0b100:])
	# Get total prob of measuring |1> on q2
	prob_real_true = np.sum(realdisc_probs[0b100:])
	cost = prob_fake_true - prob_real_true
	return cost

# We prepare the initial points of the optimization as tensorflow Variables.
init_gen_params = np.random.uniform(low=-np.pi,
                                    high=np.pi,
                                    size=(THETA_G_DIMS,))
init_disc_params = np.random.uniform(low=-np.pi,
                                     high=np.pi,
                                     size=(THETA_D_DIMS,))
gen_params = tf.Variable(init_gen_params)
disc_params = tf.Variable(init_disc_params)

# Initialize Adam optimizer from Keras. 
# Two Optimizers will compete against each other in a zero-sum-game. 
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)


# Initialize variables to track metrics while training
best_gen_params = tf.Variable(init_gen_params)
g_loss, d_loss, kl_div = [], [], []
max_iter = 3


TABLE_HEADER = "Epoch | Generator cost | Discriminator cost | KL Div. |"
print(TABLE_HEADER)

# Start Training:
for epoch in range(max_iter):
	"""
		Heuristic methods yield that five-to-one ratio between discriminator
		and generator training delivers the best results. In these steps the 
		implementation of the cost function is also embedded.
	"""
	D_STEPS = 5
	G_STEPS = 1

	for disc_training_step in range(D_STEPS):
		"""
			Disc Cost Function: E[Real_Disc_OP] - E[Gen_Disc_OP].
			We are trying to minimize this function.

			Note: E[.] = -1 corresponds to last qubit being measured 
			as 1 (classification=real) and E[.] = 1 corresponds to
			the last qubit being measured as 0 (class. = fake). 
		"""

		grad_dcost_fake = disc_fake_qnn.backward(gen_params,disc_params)[1][0,0]
		grad_dcost_real = disc_real_qnn.backward(gen_params,disc_params)[1][0,0]
		grad_dcost = grad_dcost_real - grad_dcost_fake

		discriminator_optimizer.apply_gradients(zip([grad_dcost], [disc_params]))

		if disc_training_step % D_STEPS == 0:
			d_loss.append(discriminator_cost(disc_params))

	for gen_training_step in range(G_STEPS):
		"""
			Gen Cost Function: E[Gen_Disc_OP].
			We are trying to minimize this function.
		"""

		grad_gcost = gen_qnn.backward(disc_params,gen_params)[1][0,0]
		generator_optimizer.apply_gradients(zip([grad_gcost], [gen_params]))

		if gen_training_step % G_STEPS == 0:
			g_loss.append(generator_cost(gen_params))

	"""
		Check the KL divergence of the generated state and the real data
		with the updated gen_params. If it is a new best score, update the
		best_gen_params. 
	"""
	gen_checkpoint_circuit = generator.bind_parameters(gen_params.numpy())
	gen_prob_dict = Statevector(gen_checkpoint_circuit).probabilities_dict()

	curr_kl_div = calculate_kl_div(gen_prob_dict, real_prob_dict)
	kl_div.append(curr_kl_div)
	best_kl = np.min(kl_div)

	if best_kl == curr_kl_div:
		best_gen_params = pickle.loads(pickle.dumps(gen_params))

	if epoch % 30 == 0:
		for header, val in zip(TABLE_HEADER.split('|'), (epoch, g_loss[-1], d_loss[-1],
						kl_div[-1], best_kl == curr_kl_div)):
			print(f"{val:.3g} ".rjust(len(header)), end="|")
		print()

"""
	Visualize the results.
"""
import matplotlib.pyplot as plt
fig, (loss, kl) = plt.subplots(2, sharex=True,
								gridspec_kw={'height_ratios': [0.75, 1]},
								figsize=(6,4))
fig.suptitle('QGAN training stats')
fig.supxlabel('Training step')
loss.plot(range(len(gloss)), gloss, label="Generator loss")
loss.plot(range(len(dloss)), dloss, label="Discriminator loss",
			color="C3")
loss.legend()
loss.set(ylabel='Loss')
kl.plot(range(len(kl_div)), kl_div, label="KL Divergence (zero is best)",
			color="C1")
kl.set(ylabel='KL Divergence')
kl.legend()
fig.tight_layout();

fig.savefig("QGAN_GAUSSIAN.png")