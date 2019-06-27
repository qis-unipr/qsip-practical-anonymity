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

if len(argv) > 2:
    given_fidelity = float(argv[2])
if len(argv) > 1:
    nodes = int(argv[1])

try:
    print('GHZ state fidelity calculator')

    for iteration in range(2000):
        qr = QuantumRegister(nodes)
        cr = ClassicalRegister(nodes)

        #Generates the GHZ Circuit
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

        current_fidelity = 1
        rotation_sequence = []
        vals = [ x for x in range(nodes) ]
        
        target_density_matrix = None
        state_vector = None

        # useful informations
        # resulting_density_matrix = None
        # new_rotations = [ [0 for _ in range(nodes)] for _ in range(2)]

        # calculates angles in such a way that the fidelity of the noisy GHZ is higher than the given threshold
        while current_fidelity > given_fidelity:
            fidelity = current_fidelity

            ex_q = choice(vals)
            axis = randint(0,1)
            
            
            # useful informations : saves the rotations of the qubits in a list
            # resulting_state_vector = state_vector
            # resulting_density_matrix = target_density_matrix
            # rotations = deepcopy(new_rotations)
            # new_rotations[axis][ex_q] += rotation_step
            

            if axis == 0:
                ghzstate.ry(rotation_step, qr[ex_q])
                rotation_sequence.append( (ex_q, 0) )    
            else:
                ghzstate.rx(rotation_step, qr[ex_q])
                rotation_sequence.append( (ex_q, 1) )

            # gets the state vector of the modified qubits and calculates the fidelity 
            # with respect to the perfect GHZ
            result = execute(ghzstate, backend_sim).result()
            state_vector = np.array(result.get_statevector(ghzstate))
            state_vector_reshaped = state_vector.reshape((1,-1))
            target_density_matrix = np.dot(state_vector_reshaped.transpose(), state_vector_reshaped)

            # calculates fidelity
            current_fidelity = state_fidelity(pure_density_matrix, target_density_matrix) 

        # prints useful informations
        #print('rotations per axis', rotations)
        #print('resulting state vector', resulting_state_vector)
        print(iteration, 'Fidelity =', fidelity)

        with open('output'+str(nodes)+'_{0:2.0f}'.format(given_fidelity*100)+'.txt','a') as f:
            print(str(fidelity)+';'+ str(rotation_sequence), file = f)            

except QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))