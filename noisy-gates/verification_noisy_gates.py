
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
from qiskit.circuit import Gate

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
given_fidelity = 0.9
nodes = 3
rotation_step = (2*pi)/256
ITERATIONS = 128*5

timestamp = datetime.now().strftime('%Y%m%d%H%M')

def xorBitByBit(bits):
    if( len(bits) == 0):
        return -1

    result = bits[0]
    for idx in range(1, len(bits)):
        result = result^bits[idx]
    return int(Bits(bin=bin(result)).bin)


def main():
    # circuit creates a ghz without measuring qubits.
    # this is the circuit that we will use in order to calculate fidelity
    # the idenity gate is used in order to add the depolarizing channel noise to the state
    qr = QuantumRegister(nodes)
    cr = ClassicalRegister(nodes)
    circuit = QuantumCircuit(qr, cr, name='ghz_circuit')

    # Creates the first subcircuit that represents the gate H-I applied to
    # two qubits
    sub_q_hi = QuantumRegister(nodes)
    HI_circuit = QuantumCircuit(sub_q_hi, name='hi')
    for i in range(nodes-1):
        HI_circuit.h(sub_q_hi[i])
    HI_circuit.iden(sub_q_hi[-1])

    # Convert to a gate and stick it into an arbitrary place in the bigger circuit
    HI_instr = HI_circuit.to_instruction()
    circuit.append(HI_instr, [qr[i] for i in range(nodes)])

    for i in range(nodes-1):
        circuit.cx(qr[i], qr[-1])

    # Creates the second subcircuit that represents the gate H-H applied to
    # two qubits
    sub_q_hh = QuantumRegister(nodes)
    HH_circuit = QuantumCircuit(sub_q_hh, name='hh')
    for i in range(nodes):
        HH_circuit.h(sub_q_hh[i])

    HH_instr = HH_circuit.to_instruction()
    circuit.append(HH_instr, [qr[i] for i in range(nodes)])

    statevector_simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, statevector_simulator, shots = 1000).result()
    state = result.get_statevector(circuit)
    #print('ghz statevector\n',state)

    # in order to calculate the probability in the depolarizing channel, we have to
    # calculate first the noisy rho density matrix
    psi = np.array(state)
    psi_density_matrix = psi.reshape((1,-1))
    psi_density_matrix = np.dot(psi_density_matrix.transpose(), psi_density_matrix)

    p = 0.00

    ket_0 = np.matrix([[1,0]], dtype = np.int16).T
    psi_0 = ket_0
    for _ in range(nodes-1):
        psi_0 = np.kron(psi_0, ket_0)
    rho_0 = np.outer(psi_0.T, psi_0)

    ket_1 = np.matrix([[0,1]], dtype = np.complex64).T
    psi_1 = ket_1
    for _ in range(nodes-1):
        psi_1 = np.kron(psi_1, ket_1)

    H = (1/(2**(1/2)))*np.matrix(([1, 1], [1, -1]), dtype = np.complex64)
    I = np.identity(2)
    X = np.matrix(([0, 1], [1, 0]), dtype = np.complex64)
    I_noise = np.identity(2**nodes)/(2**nodes)

    HkI = H.copy()
    for _ in range(nodes-2):
        HkI = np.kron(HkI, H)
    HkI = np.kron(HkI, I)

    rho_1 = np.dot(HkI, rho_0)
    rho_1 = np.dot(rho_1, HkI.getH())
    rho = psi_density_matrix.copy()

    dm_0 = np.outer(ket_0, ket_0.T)
    dm_1 = np.outer(ket_1, ket_1.T)

    # creates a list of cnots. The target is always the last one
    cnot_list = []
    for idx in range(nodes-1):
        f_part = dm_0.copy()
        s_part = dm_1.copy()
        for _ in range(idx):
            f_part = np.kron(np.identity(2), f_part)
            s_part = np.kron(np.identity(2), s_part)
        for _ in range(idx, nodes-1):
            f_part = np.kron(f_part, np.identity(2))
        for _ in range(idx, nodes-2):
            s_part = np.kron(s_part, np.identity(2))
        s_part = np.kron(s_part, X)
        cnot_list.append(f_part+s_part)

    actual_fidelity = 1
    fidelity = 1
    resulting_p = p
    resulting_density_matrix = rho
    while actual_fidelity > given_fidelity:
        resulting_p = p
        fidelity = actual_fidelity
        resulting_density_matrix = rho

        p += 0.00001
        # apply the noise (general function): first stage
        rho = p*I_noise + (1-p)*rho_1

        for cnot in cnot_list:
            rho = np.dot(cnot, rho)
            rho = np.dot(rho, cnot.getH())

            # for every cnot, apply noise
            rho = p*I_noise + (1-p)*rho

        HkH = H.copy()
        for _ in range(nodes-1):
            HkH = np.kron(HkH, H)

        rho = np.dot(HkH, rho)
        rho = np.dot(rho, HkH.getH())

        rho = p*I_noise + (1-p)*rho
        actual_fidelity = state_fidelity(rho, psi_density_matrix)

    print('depolarizing probability:', resulting_p)
    print('Additional infos:',
        '\n- trace:', np.trace(resulting_density_matrix),
        '\n- fidelity:', fidelity)

    #################### code that models the noise ####################################
    # Once we have the probability that gives us a specific fidelity in the depolarizing
    # channel, we simulate the circuit.
    # Creates and adds depolarizing error to specific qubit gates
    noise_model = NoiseModel()
    error = depolarizing_error(resulting_p, nodes)
    error2 = depolarizing_error(resulting_p, 2)
    noise_model.add_all_qubit_quantum_error(error, ['hi','hh'])
    noise_model.add_all_qubit_quantum_error(error2, 'cx')
    print(noise_model)
    print('Running verification protocol...')

    # creates a dictionary that counts the results
    angles_dict = {}
    for b in range(2**nodes):
        angles_dict[bin(b)[2:].zfill(nodes)] = 0

    for iteration in range(ITERATIONS):
        qr = QuantumRegister(nodes)
        cr = ClassicalRegister(nodes)
        circuit = QuantumCircuit(qr, cr, name='ghz_circuit')

        qr = QuantumRegister(nodes)
        cr = ClassicalRegister(nodes)
        circuit = QuantumCircuit(qr, cr, name='ghz_circuit')

        # Creates the first subcircuit that represents the gate H-I applied to
        # two qubits
        sub_q_hi = QuantumRegister(nodes)
        HI_circuit = QuantumCircuit(sub_q_hi, name='hi')
        for i in range(nodes-1):
            HI_circuit.h(sub_q_hi[i])
        HI_circuit.iden(sub_q_hi[-1])

        # Convert to a gate and stick it into an arbitrary place in the bigger circuit
        HI_instr = HI_circuit.to_instruction()
        circuit.append(HI_instr, [qr[i] for i in range(nodes)])

        for i in range(nodes-1):
            circuit.cx(qr[i], qr[-1])

        # Creates the second subcircuit that represents the gate H-H applied to
        # two qubits
        sub_q_hh = QuantumRegister(nodes)
        HH_circuit = QuantumCircuit(sub_q_hh, name='hh')
        for i in range(nodes):
            HH_circuit.h(sub_q_hh[i])

        HH_instr = HH_circuit.to_instruction()
        circuit.append(HH_instr, [qr[i] for i in range(nodes)])


        DEBUG = timestamp+'_'
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

        DEBUG += '_noisy_gates_'+str(nodes)

        random_angles_steps = random_angles[:]
        random_angles[:] = np.array(random_angles) * rotation_step

        for node in range(nodes):
            circuit.rz(-random_angles[node], qr[node])
            circuit.ry(-pi/2, qr[node])

        circuit.measure(qr, cr)

        result = execute(circuit,
                        backend = Aer.get_backend('qasm_simulator'),
                        shots = 1,
                        basis_gates = noise_model.basis_gates,
                        noise_model = noise_model).result()

        counts_device = result.get_counts(circuit)

        with open(folder_name+'/'+DEBUG+'.txt','a') as f:
            string = ''
            for key in counts_device:
                angles_dict[key] += 1
                string += (key+','+str(sum(random_angles_steps)))
                if xorBitByBit([int(elem) for elem in key]) == int(sum(random_angles_steps)/128)%2:
                    string += ',ok'
            print('{0:4.0f}'.format(fidelity*1000)+','+str(ITERATIONS)+'/'+str(iteration+1)+','+string, file = f)

    with open(folder_name+'/'+DEBUG+'.txt','a') as f:
        print("--- %s seconds ---" % (time.time() - start_time), file = f)

    plot_histogram(angles_dict, title='Verification protocol results').savefig(folder_name+'/'+DEBUG+'.png')

if __name__ == "__main__":
    np.set_printoptions(suppress = True,linewidth=np.inf)

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
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)
    start_time = time.time()
    main()
