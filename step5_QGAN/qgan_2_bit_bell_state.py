"""
	This project implements a Quantum Generative Adversarial Network to
	train a 2 qubit, first Bell state (|00> + |11>). Discriminator is 
	also implemented with a Quantum Ansatz.
	Uses the ADAM optimizer from tensorflow's Keras. Therefore we convert
	the theta values to tensorflow's Variable class.

	cost_discriminator__theta_g(theta_d) = Pr[gen_disc == real] - Pr[disc_real == real]
	cost_generator__theta_d(theta_g) = -Pr[gen_disc == real]
"""

from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import  ParameterVector
from qiskit.quantum_info import Statevector

import tensorflow as tf
import numpy as np
import pickle

qi_sv = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))

# Real Data, that is aimed to be replicated is the first, 2-qubit Bell State 
real_data = QuantumCircuit(2)
real_data.h(0)
real_data.cx(0,1)

# Create the Ansatz for Generator and Discriminator
generator = TwoLocal(
		2, ['ry', 'rz'], 'cz', 'full', reps=2, parameter_prefix='θ_g', name='Generator'
	)
generator = generator.decompose()

discriminator = QuantumCircuit(3, name="Discriminator")
disc_weights = ParameterVector('θ_d', 12)
discriminator.barrier()
discriminator.h(0)
discriminator.rx(disc_weights[0], 0)
discriminator.ry(disc_weights[1], 0)
discriminator.rz(disc_weights[2], 0)
discriminator.rx(disc_weights[3], 1)
discriminator.ry(disc_weights[4], 1)
discriminator.rz(disc_weights[5], 1)
discriminator.rx(disc_weights[6], 2)
discriminator.ry(disc_weights[7], 2)
discriminator.rz(disc_weights[8], 2)
discriminator.cx(0, 2)
discriminator.cx(1, 2)
discriminator.rx(disc_weights[9], 2)
discriminator.ry(disc_weights[10], 2)
discriminator.rz(disc_weights[11], 2)


N_GPARAMS = generator.num_parameters
N_DPARAMS = discriminator.num_parameters


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


# Generator + Discriminator Circuit:
gen_disc_circuit = QuantumCircuit(3)
gen_disc_circuit.compose(generator, inplace=True)
gen_disc_circuit.compose(discriminator,inplace=True)

# Real data fed Discriminator Circuit:
real_disc_circuit = QuantumCircuit(3)
real_disc_circuit.compose(real_data, inplace=True)
real_disc_circuit.compose(discriminator, inplace=True)


# Instantiate three models for optimization
gen_qnn = CircuitQNN(
		gen_disc_circuit, 
		gen_disc_circuit.parameters[:N_DPARAMS], # Frozen params: θ_d
		gen_disc_circuit.parameters[N_DPARAMS:], # Variational params: θ_g
		sparse=True,
		quantum_instance=qi_sv
	)

disc_fake_qnn = CircuitQNN(
		gen_disc_circuit,
		gen_disc_circuit.parameters[N_DPARAMS:], # Frozen params: θ_g
		gen_disc_circuit.parameters[:N_DPARAMS], # Variational params: θ_d
		sparse=True,
		quantum_instance=qi_sv
	)

disc_real_qnn = CircuitQNN(
		real_disc_circuit,
		[],
		gen_disc_circuit.parameters[:N_DPARAMS], # θ_d gets trained on the real data circuit
		sparse=True,
		quantum_instance=qi_sv
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
                                    size=(N_GPARAMS,))
init_disc_params = np.random.uniform(low=-np.pi,
                                     high=np.pi,
                                     size=(N_DPARAMS,))
gen_params = tf.Variable(init_gen_params)
disc_params = tf.Variable(init_disc_params)


"""
	We initialize the generator circuit. 
	Visualize the initial generator distribution.
"""

init_gen_circuit = generator.bind_parameters(init_gen_params)
init_prob_dict = Statevector(init_gen_circuit).probabilities_dict()

import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram

fig, ax1 = plt.subplots(1, 1, sharey=True)
ax1.set_title("Initial generator distribution")
plot_histogram(init_prob_dict, ax=ax1)
fig.savefig("results_2_bit_bell/initial generator distribution")


# Initialize Adam optimizer from Keras. 
# Two Optimizers will compete against each other in a zero-sum-game. 
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

# Initialize variables to track metrics while training
best_gen_params = tf.Variable(init_gen_params)
g_loss = []
d_loss = []
kl_div = []
max_iter = 100


TABLE_HEADER = "Epoch | Generator cost | Discriminator cost | KL Div. |"
print(TABLE_HEADER)

for epoch in range(max_iter):
	"""
		Heuristic methods yield that five-to-one ratio between discriminator
		and generator training delivers the best results. In these steps the 
		implementation of the cost function is also embedded.
	"""
	D_STEPS = 5
	G_STEPS = 1

	for disc_training_step in range(D_STEPS):
		d_fake_matrix = disc_fake_qnn.backward(gen_params.numpy(), disc_params.numpy())[1].todense()[0][0b100:]

		"""
		This step delivers the a matrix:
		[
			[d_theta_d1(<000|phi(theta)>),d_theta_d1(<001|phi(theta)>),d_theta_d1(<010|phi(theta)>)...],
			[d_theta_d2(<000|phi(theta)>),d_theta_d2(<001|phi(theta)>),d_theta_d2(<010|phi(theta)>)...],
			...
			[d_theta_d12(<000|phi(theta)>),d_theta_d12(<001|phi(theta)>),d_theta_d12(<010|phi(theta)>)...],
		]
		of which we slice the colums after (including) 0b100 state -> which corresponds to
		all the states that result in discriminator classifying as real.
		The derivatives are all evaluated at the thata = [gen_params,disc_params] values.
		"""

		grad_d_fake = np.sum(d_fake_matrix, axis=0)
		# Here we add up every column of a row and get at the end a column vector.


		d_real_matrix = disc_real_qnn.backward([], disc_params)[1].todense()[0][0b100:]
		grad_d_real = np.sum(d_real_matrix, axis=0)

		# Calculate the grad(cost_disc) = grad(Pr[fake_data = real] - Pr[real_data = real])
		# and conver it to tensorflow tensor.
		grad_d_cost = [grad_d_fake[i] - grad_d_real[i] for i in range(N_DPARAMS)]
		grad_d_cost = tf.convert_to_tensor(grad_d_cost)

		discriminator_optimizer.apply_gradients(zip([grad_d_cost], [disc_params]))

		if disc_training_step % D_STEPS == 0:
			d_loss.append(discriminator_cost(disc_params))

	for gen_training_step in range(G_STEPS):
		g_matrix = gen_qnn.backward(disc_params.numpy(),gen_params.numpy())[1].todense()[0][0b100:]
		grad_g = np.sum(g_matrix, axis=0)

		# Gradient of Cost function of the generator: -Pr[gen_disc == real]
		grad_g_cost = -grad_g
		grad_g_cost = tf.convert_to_tensor(grad_g_cost)

		generator_optimizer.apply_gradients(zip([grad_g_cost], [gen_params]))
		g_loss.append(generator_cost(gen_params))

	# Calculate the KL-div of the generated state and real state
	gen_checkpoint_circuit = generator.bind_parameters(gen_params.numpy())
	generated_state = Statevector(gen_checkpoint_circuit).probabilities_dict()
	real_state = Statevector(real_data).probabilities_dict()
	current_kl = calculate_kl_div(generated_state, real_state)
	kl_div.append(current_kl)

	# If a new best theta_g is found, we deserialize and reserialize 
	# to make sure there are no zero links
	if np.min(kl_div) == current_kl:
		best_gen_params = pickle.loads(pickle.dumps(gen_params))

	if epoch % 10 == 0:
		for header, val in zip(TABLE_HEADER.split('|'),
				(epoch, g_loss[-1], d_loss[-1], kl_div[-1])):
			print(f"{val:.3g} ".rjust(len(header)), end="|")

"""
	Visualize the results. One plot for losses of discriminator and 
	generator and one for the KL divergence over the training steps.
"""

fig, (loss, kl) = plt.subplots(2, sharex=True,
                               gridspec_kw={'height_ratios': [0.75, 1]},
                               figsize=(6,4))
fig.suptitle('QGAN training stats')
fig.supxlabel('Training step')
loss.plot(range(len(g_loss)), g_loss, label="Generator loss")
loss.plot(range(len(d_loss)), d_loss, label="Discriminator loss",
          color="C3")
loss.legend()
loss.set(ylabel='Loss')
kl.plot(range(len(kl_div)), kl_div, label="KL Divergence",
        color="C1")
kl.set(ylabel='KL Divergence')
kl.legend()
fig.tight_layout()
fig.savefig('results_2_bit_bell/QGAN_training_chart.png')


"""
	Visualize the trained generator distribution.
"""
gen_checkpoint_circuit = generator.bind_parameters(
    best_gen_params.numpy())
gen_prob_dict = Statevector(gen_checkpoint_circuit).probabilities_dict()
real_prob_dict = Statevector(real_data).probabilities_dict() # constant
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
plot_histogram(gen_prob_dict, ax=ax1)
ax1.set_title("Trained generator distribution")
plot_histogram(real_prob_dict, ax=ax2)
ax2.set_title("Real distribution")
fig.tight_layout()
fig.savefig('results_2_bit_bell/trained_generator_state.png')
