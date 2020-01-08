import sys
import numpy as np
from sys import argv
import os
import logging

# Import the QISKit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, qasm, QiskitError
from qiskit import execute, BasicAer

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


def apply_rotations(quantum_register, classical_register, circuit, angles, qubit_to_rotate):
    # operazioni del verification protocol
    circuit.rz(-angles[qubit_to_rotate], quantum_register[qubit_to_rotate])
    circuit.ry(-pi/2, quantum_register[qubit_to_rotate])

    # state vector prima della misura
    result = execute(circuit, backend_sim).result()
    state_vector = np.array(result.get_statevector(circuit))
    print('qubit to rotate:', qubit_to_rotate, 'state after rotation:\n', state_vector.reshape(-1, 1), '\n')

    # state vector post misura
    circuit.measure(quantum_register[qubit_to_rotate], classical_register[qubit_to_rotate])
    result = execute(circuit, backend_sim).result()
    state_vector = np.array(result.get_statevector(circuit))
    print('qubit to rotate:', qubit_to_rotate, 'state after measurement:\n', state_vector.reshape(-1, 1), '\n')


try:
    print('GHZ state variation during Verification Protocol')
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
    state_vector = np.array(result.get_statevector(ghzstate))

    print('ghz:\n', state_vector.reshape(-1,1), '\n')

    random_angles = [-1]
    while sum(random_angles) % 128 != 0:
        random_angles = list([randint(0, 127) for _ in range(nodes)])

    print('sum random angles', sum(random_angles))
    random_angles[:] = np.array(random_angles) * rotation_step

    for node in range(nodes):
        apply_rotations(qr, cr, ghzstate, random_angles, node)

except QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
