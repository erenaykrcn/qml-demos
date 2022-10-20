"""
    Script to show the Barren Plateau Problem for growing number of qubits.
    Plateau refers to the exponential decrease of the variance of the gradient
    of the cost function as number of qubits increase.
"""

import numpy as np
from qiskit.algorithms.optimizers import GradientDescent, SPSA, QNSPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import Z, I, StateFn, PauliExpectation, Gradient, NaturalGradient
from qiskit.opflow import PauliExpectation, CircuitSampler


quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator'),
                                   # we set a seed for reproducibility
                                   shots = 800, seed_simulator = 2718,
                                   seed_transpiler = 2718)
sampler = CircuitSampler(quantum_instance, caching="all")


def sample_gradients(num_qubits, reps, local=False):
    """Sample the gradient of our model for ``num_qubits`` qubits and
    ``reps`` repetitions.

    We sample 100 times for random parameters and compute the gradient
    of the first RY rotation gate.
    """
    index = num_qubits - 1

    # local or global operator
    if local:
        operator = Z ^ Z ^ (I ^ (num_qubits - 2))
    else:
        operator = Z ^ num_qubits

    # real amplitudes ansatz
    ansatz = RealAmplitudes(num_qubits, entanglement='linear', reps=reps)

    # construct Gradient we want to evaluate for different values
    expectation = StateFn(operator,
                          is_measurement=True).compose(StateFn(ansatz))
    grad = Gradient().convert(expectation,
                              params=ansatz.parameters[index])

    # evaluate for 100 different, random parameter values
    num_points = 100
    grads = []
    for _ in range(num_points):
        # points are uniformly chosen from [0, pi]
        point = np.random.uniform(0, np.pi, ansatz.num_parameters)
        value_dict = dict(zip(ansatz.parameters, point))
        grads.append(sampler.convert(grad, value_dict).eval())

    return grads


num_qubits = list(range(2, 8))

# Linear depth, Global Operator Gradients
gradients_1 = [sample_gradients(n, n) for n in num_qubits]

# Constant depth, Global Operator Gradients
gradients_2 = [sample_gradients(n, 1) for n in num_qubits]

# Linear depth, Local Operator Gradients
gradients_3 = [sample_gradients(n, n, True) for n in num_qubits]

# Constant depth, Local Operator Gradients
gradients_4 = [sample_gradients(n, 1, True) for n in num_qubits]


"""
    Visualization of the Variance of the sample gradients.
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 3))
plt.semilogy(num_qubits, np.var(gradients_1, axis=1), 'o-', label='Linear depth, Global Operator')
plt.semilogy(num_qubits, np.var(gradients_2, axis=1), 'o-', label='Constant depth, Global Operator')
plt.semilogy(num_qubits, np.var(gradients_3, axis=1), 'o-', label='Linear depth, Local Operator')
plt.semilogy(num_qubits, np.var(gradients_4, axis=1), 'o-', label='Constant depth, Local Operator')
plt.xlabel('number of qubits')
plt.ylabel(r'$\mathrm{Var}[\partial_{\theta 1}\langle E(\theta)\rangle]$')
plt.legend()
plt.savefig("variance_to_num_of_qubits.png")