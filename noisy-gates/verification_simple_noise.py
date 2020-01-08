
# This code works with qiskit 0.12

import sys
import numpy as np
import os
import logging

# Import the QISKit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QiskitError
from qiskit import execute, Aer

# useful additional packages
from qiskit.providers.aer import noise
from qiskit.quantum_info import state_fidelity
from qiskit.visualization import plot_histogram

# utilities
from math import pi
from random import randint
from sys import argv
from datetime import datetime
from bitstring import BitArray, Bits

import time

# Qiskit Aer noise module imports
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise.errors import depolarizing_error

logging.getLogger('qiskit._compiler').setLevel(logging.INFO)
logging.getLogger('qiskit.mapper._mapping').setLevel(logging.DEBUG)

# constants
even = None
folder_name = 'results'
shots = 500
rotation_step = (2*pi)/256
given_fidelity = 0.9
nodes = 3
ITERATIONS = 128*5

timestamp = datetime.now().strftime('%Y%m%d%H%M')

def xorBitByBit(bits):
    if len(bits) == 0:
        return -1

    result = bits[0]
    for idx in range(1, len(bits)):
        result = result ^ bits[idx]
    return int(Bits(bin=bin(result)).bin)

if len(argv) > 1:
    if argv[1] == 'e':
        even = True
    elif argv[1] == 'o':
        even = False
    else:
        print('please specify even or odd simulation!')
        exit()
if len(argv) > 2:
    nodes = int(argv[2])
if len(argv) > 3:
    given_fidelity = float(argv[3])

if not os.path.exists('./'+folder_name):
    path = './'+folder_name
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
start_time = time.time()

try:
    # circuit creates a ghz without measuring qubits.
    # this is the circuit that we will use in order to calculate fidelity
    # the idenity gate is used in order to add the depolarizing channel noise to the state
    qr = QuantumRegister(nodes)
    cr = ClassicalRegister(nodes)
    circuit = QuantumCircuit(qr, cr, name='ghz_circuit')

    for i in range(nodes-1):
        circuit.h(qr[i])
    for i in range(nodes-1):
        circuit.cx(qr[i], qr[-1])
    for i in range(nodes):
        circuit.h(qr[i])

    for i in range(nodes):
        circuit.iden(qr[i])

    statevector_simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, statevector_simulator, shots=shots).result()
    state = result.get_statevector(circuit)
    # print('ghz statevector\n',state)

    # in order to calculate the probability in the depolarizing channel, we have to
    # calculate first the noisy rho density matrix
    psi = np.array(state)
    psi_density_matrix = psi.reshape((1, -1))
    psi_density_matrix = np.dot(psi_density_matrix.transpose(), psi_density_matrix)
    # print('psi density matrix \n', psi_density_matrix)

    actual_p = 0.00
    rho_density_matrix = psi_density_matrix.copy()
    actual_fidelity = state_fidelity(rho_density_matrix, psi_density_matrix)

    while actual_fidelity > given_fidelity:
        p = actual_p
        resulting_density_matrix = rho_density_matrix.copy()
        fidelity = actual_fidelity

        actual_p += 0.001
        rho_density_matrix = actual_p*(np.identity(len(state))/(2**nodes)) + (1 - actual_p)*psi_density_matrix
        rho_density_matrix = rho_density_matrix/np.trace(rho_density_matrix)
        actual_fidelity = state_fidelity(rho_density_matrix, psi_density_matrix)

    # rho_density_matrix = (actual_p*np.identity(len(state)))/nodes + (1 - actual_p)*psi_density_matrix
    # fidelity = state_fidelity(rho_density_matrix, psi_density_matrix)
    print('depolarizing probability:', p)
    print('resulting fidelity:', fidelity)
    # print('resulting rho density matrix:\n', resulting_density_matrix)
    # print('tizio:', np.trace(resulting_density_matrix), p**(1/nodes))

    # code that models the noise
    # Once we have the probability that gives us a specific fidelity in the depolarizing
    # channel, we simulate the circuit.
    # Creates and adds depolarizing error to specific qubit gates
    noise_model = NoiseModel()
    error = depolarizing_error(p**(1/nodes), 1)
    noise_model.add_all_qubit_quantum_error(error, 'id')
    print(noise_model)

    # The ghzstate circuit is used in order to verify the effect of the noise on the
    # ghz state. This part can be commented.
    '''ghzstate = QuantumCircuit(qr, cr, name='ghz')

    for i in range(nodes):
        ghzstate.iden(qr[i])

    for i in range(nodes-1):
        ghzstate.h(qr[i])

    for i in range(nodes-1):
        ghzstate.cx(qr[i], qr[-1])

    for i in range(nodes):
        ghzstate.h(qr[i])

    ghzstate.measure(qr, cr)

    job = execute(ghzstate, Aer.get_backend('qasm_simulator'))
    result_ideal = job.result()
    print('ideal ghz',result_ideal.get_counts(0))
    plot_histogram(result_ideal.get_counts(0)).savefig('ideal.png')

    job = execute(ghzstate, Aer.get_backend('qasm_simulator'),
              basis_gates=noise_model.basis_gates,
              noise_model=noise_model)
    result_noise_model = job.result()
    counts_noise_model = result_noise_model.get_counts(0)
    print('noisy ghz', counts_noise_model)
    # Plot noisy output
    plot_histogram(counts_noise_model).savefig('noisy.png')'''

    print('Running verification protocol...')
    # creates a dictionary that counts the results
    angles_dict = {}
    for b in range(2**nodes):
        angles_dict[bin(b)[2:].zfill(nodes)] = 0

    for iteration in range(ITERATIONS):
        qr = QuantumRegister(nodes)
        cr = ClassicalRegister(nodes)
        circuit = QuantumCircuit(qr, cr, name='ghz_circuit')

        for i in range(nodes-1):
            circuit.h(qr[i])
        for i in range(nodes-1):
            circuit.cx(qr[i], qr[-1])
        for i in range(nodes):
            circuit.h(qr[i])

        for i in range(nodes):
            circuit.iden(qr[i])

        DEBUG = timestamp + '_'
        random_angles = [-1]
        if even:
            DEBUG += 'even'
            while (sum(random_angles) / 128) % 2 != 0:
                random_angles = list([randint(0, 127) for _ in range(nodes-1)])
                if iteration % 128 == 1 or iteration % 128 == 2:
                    random_angles.append(3)
                else:
                    random_angles.append(iteration%128)

        else:
            DEBUG += 'odd'
            while (sum(random_angles) / 128) % 2 != 1:
                random_angles = list([randint(0, 127) for _ in range(nodes-1)])
                random_angles.append(iteration%128)

        DEBUG += '_I_gates_b_'+str(nodes)

        # print('sum random angles', sum(random_angles))
        random_angles_steps = random_angles[:]
        random_angles[:] = np.array(random_angles) * rotation_step

        for node in range(nodes):
            circuit.rz(-random_angles[node], qr[node])
            circuit.ry(-pi/2, qr[node])

        circuit.measure(qr, cr)

        result = execute(circuit,
                        backend=Aer.get_backend('qasm_simulator'),
                        shots=1,
                        basis_gates=noise_model.basis_gates,
                        noise_model=noise_model).result()

        counts_device = result.get_counts(circuit)

        with open(folder_name+'/'+DEBUG+'.txt','a') as f:
            string = ''
            for key in counts_device:
                angles_dict[key] += 1
                string += (key+','+str(sum(random_angles_steps)))
                if xorBitByBit([int(elem) for elem in key]) == int(sum(random_angles_steps)/128)%2:
                    string += ',ok'
            print('{0:4.0f}'.format(fidelity*1000)+','+str(ITERATIONS)+'/'+str(iteration+1)+','+string, file=f)

    with open(folder_name + '/' + DEBUG + '.txt', 'a') as f:
        print("--- %s seconds ---" % (time.time() - start_time), file = f)

    plot_histogram(angles_dict, title='Verification protocol results').savefig(folder_name + '/' + DEBUG + '.png')

except QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
