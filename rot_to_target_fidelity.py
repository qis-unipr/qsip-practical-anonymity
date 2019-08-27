import sys
import numpy as np
from sys import argv
import os
import logging

# Import the QISKit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, qasm, QiskitError
from qiskit import execute, BasicAer

# useful additional packages
from qiskit.quantum_info import state_fidelity
from qiskit.tools.visualization import plot_state_city

# utilities
from copy import deepcopy
from math import pi
from random import choice, randint
import time
from ast import literal_eval

logging.getLogger('qiskit._compiler').setLevel(logging.INFO)
logging.getLogger('qiskit.mapper._mapping').setLevel(logging.DEBUG)

# constants
rotation_step = (2*pi)/256
given_fidelity = 0.9
nodes = 3

# partendo da un file con una determinata fidelity porta elimina le ultime rotazioni
# fino al raggiungimento di un altro valore desiderato
# utilizzato a scopo di DEBUG
try:
    print('GHZ state fidelity calculator')
    qr = QuantumRegister(nodes)
    cr = ClassicalRegister(nodes)

    ghzstate = QuantumCircuit(qr, cr, name='ghzstate')
    for i in range(nodes-1):
        ghzstate.h(qr[i])
        ghzstate.cx(qr[i], qr[-1])

    for i in range(nodes):
        ghzstate.h(qr[i])
    
    backend_sim = BasicAer.get_backend('statevector_simulator')
    result = execute(ghzstate, backend_sim).result()

    pure_state_vector = np.array(result.get_statevector(ghzstate))
    pure_state_vector = pure_state_vector.reshape((1,-1))
    pure_density_matrix = np.dot(pure_state_vector.transpose(), pure_state_vector)

    with open('results/test_debug_3.csv','r') as f:
        lines = f.readlines()
    
    rotations = lines[5].split(';')[1:][0]
    rotations = literal_eval(rotations)

    qubits, rotations = [ list(tup) for tup in zip(*rotations)]

    F_fit = 1

    for idx, q in enumerate(qubits):
        fidelity = F_fit
        if rotations[idx] == 0:
            ghzstate.ry(rotation_step, qr[q])  
        else:
            ghzstate.rx(rotation_step, qr[q])

        result = execute(ghzstate, backend_sim).result()
        state_vector = np.array(result.get_statevector(ghzstate))
        state_vector_reshaped = state_vector.reshape((1,-1))
        
        target_density_matrix = np.dot(state_vector_reshaped.transpose(), state_vector_reshaped)

        # calculate fidelity and purity of fitted state
        F_fit = state_fidelity(pure_density_matrix, target_density_matrix) 
        if F_fit < 0.95:
            rotations = rotations[:idx]
            qubits = qubits[:idx]
            break

    print(rotations)
    print(qubits)
    print(fidelity)
    print(len(literal_eval(lines[5].split(';')[1:][0])), len(rotations),len(qubits))

    with open('test_different_fidelity.csv','w') as f:
        print(str(lines[5].split(';')[0])+';'+str(literal_eval(lines[5].split(';')[1:][0])), file = f)
        print(str(lines[5].split(';')[0])+';'+str(list(zip(qubits,rotations))), file = f)

except QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))