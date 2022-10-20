import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization.bloch import Bloch
from qiskit.quantum_info import Statevector

# First, we need to define the circuits:
theta_param = Parameter('θ')
phi_param = Parameter('Φ')

# Circuit B
qc = QuantumCircuit(1)
qc.h(0)
qc.rz(theta_param, 0)
qc.rx(phi_param, 0)

# Next we uniformly sample the parameter space for the two parameters theta and phi
np.random.seed(0)
num_param = 1000
theta = [2*np.pi*np.random.uniform() for i in range(num_param)]
phi = [2*np.pi*np.random.uniform() for i in range(num_param)]

def state_to_bloch(state_vec):
    # Converts state vectors to points on the Bloch sphere
    phi = np.angle(state_vec.data[1])-np.angle(state_vec.data[0])
    theta = 2*np.arccos(np.abs(state_vec.data[0]))
    return [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]

# Bloch sphere plot formatting
width, height = plt.figaspect(1/2)
fig=plt.figure(figsize=(width, height))
ax = fig.add_subplot(1, 2, 1, projection='3d')
b = Bloch(ax)
b.point_color = ['tab:blue']
b.point_marker = ['o']
b.point_size =[2]

# Calculate state vectors for circuit A and circuit B for each set of sampled parameters
# and add to their respective Bloch sphere
for i in range(num_param):    
    state=Statevector.from_instruction(qc.bind_parameters({theta_param:theta[i], phi_param:phi[i]}))
    b.add_points(state_to_bloch(state))

b.show()