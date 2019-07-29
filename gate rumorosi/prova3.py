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

import qiskit.tools.qcvv.tomography as tomo

# Qiskit Aer noise module imports
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise.errors import depolarizing_error

logging.getLogger('qiskit._compiler').setLevel(logging.INFO)
logging.getLogger('qiskit.mapper._mapping').setLevel(logging.DEBUG)

# constants
shots = 500
rotation_step = (2*pi)/256
given_fidelity = 0.9
nodes = 2

timestamp = datetime.now().strftime('%Y%m%d%H%M')
DEBUG = timestamp+'_odd_prova'

def xorBitByBit(bits):
    if( len(bits) == 0):
        return -1

    result = bits[0] 
    for idx in range(1, len(bits)):
        result = result^bits[idx]
    return int(Bits(bin=bin(result)).bin)

if len(argv) > 1:
    given_fidelity = float(argv[2])
if len(argv) > 2:
    nodes = int(argv[1])


np.set_printoptions( suppress = True)

try:
    # circuit creates a ghz without measuring qubits.
    # this is the circuit that we will use in order to calculate fidelity
    # the idenity gate is used in order to add the depolarizing channel noise to the state
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circuit = QuantumCircuit(qr, cr, name='ghz_circuit')  

    # Creates the first subcircuit that represents the gate H-I applied to
    # two qubits
    sub_q_hi = QuantumRegister(2)
    HI_circuit = QuantumCircuit(sub_q_hi, name='hi')
    HI_circuit.h(sub_q_hi[0])
    HI_circuit.iden(sub_q_hi[1])

    # Convert to a gate and stick it into an arbitrary place in the bigger circuit
    HI_instr = HI_circuit.to_instruction()
    circuit.append(HI_instr, [qr[0], qr[1]])

    circuit.cx(qr[0],qr[1])

    # Creates the second subcircuit that represents the gate H-H applied to
    # two qubits
    sub_q_hh = QuantumRegister(2)
    HH_circuit = QuantumCircuit(sub_q_hh, name='hh')
    HH_circuit.h(sub_q_hh[0])
    HH_circuit.h(sub_q_hh[1])

    HH_instr = HH_circuit.to_instruction()
    circuit.append(HH_instr, [qr[0], qr[1]])

    # Calculates the state vector of the ideal state
    statevector_simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, statevector_simulator, shots = shots).result()
    state = result.get_statevector(circuit)

    # In order to calculate the probability in the depolarizing channel, we have to 
    # calculate first the noisy rho density matrix
    psi = np.array(state)
    psi_density_matrix = psi.reshape((1,-1))
    psi_density_matrix = np.dot(psi_density_matrix.transpose(), psi_density_matrix)
    #print('psi density matrix \n', psi_density_matrix)

    p = 0.05

    ket_0 = np.matrix([[1,0]], dtype = np.complex64).T
    psi_0 = np.kron(ket_0,ket_0)
    rho_0 = np.outer(psi_0.T,psi_0)
    
    H = (1/(2**(1/2)))*np.matrix(([1, 1], [1, -1]), dtype = np.complex64)
    I = np.identity(2)
    I_noise = np.identity(2**nodes)/(2**nodes)

    HkI = np.kron(H, I)
    rho_1 = np.dot(HkI, rho_0)
    rho_1 = np.dot(rho_1, HkI.getH())
    rho_3n = psi_density_matrix.copy()

    actual_fidelity = 1
    while actual_fidelity > given_fidelity:
        resulting_p = p
        fidelity = actual_fidelity
        resulting_density_matrix = rho_3n

        p += 0.00001
        # apply the noise (general function)
        rho_1n = p*I_noise + (1-p)*rho_1
        cnot = np.matrix([[1,0,0,0], [0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype = np.complex64)
        
        rho_2 = np.dot(cnot, rho_1n)
        rho_2 = np.dot(rho_2, cnot.getH())

        # apply the noise (general function)
        rho_2n = p*I_noise + (1-p)*rho_2

        HcH = np.kron(H,H)
        rho_3 = np.dot(HcH, rho_2n)
        rho_3 = np.dot(rho_3, HcH.getH())

        # apply the noise (general function)
        rho_3n = p*I_noise + (1-p)*rho_3
        actual_fidelity = state_fidelity(rho_3n, psi_density_matrix)


    noise_model = NoiseModel()    
    error = depolarizing_error(p**(1/nodes), 2)
    noise_model.add_all_qubit_quantum_error(error, ['hi','hh','cx'])
    print(noise_model)

    result = execute(circuit, 
                    backend = Aer.get_backend('qasm_simulator'), 
                    shots = shots,
                    basis_gates = noise_model.basis_gates,
                    noise_model = noise_model).result()

    print('depolarizing probability:', resulting_p)
    print('resulting fidelity:', fidelity)
    print('resulting rho density matrix:\n', resulting_density_matrix)
    print('trace:', np.trace(resulting_density_matrix), 'error pb for each gate', p**(1/nodes))

    ITERATIONS = 500
    print('Running verification protocol...')
    angles_dict = {'00':0, '01':0, '10':0, '11':0}
    for iteration in range(ITERATIONS):
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr, name='ghz_circuit')  

        # Creates the first subcircuit that represents the gate H-I applied to
        # two qubits
        sub_q_hi = QuantumRegister(2)
        HI_circuit = QuantumCircuit(sub_q_hi, name='hi')
        HI_circuit.h(sub_q_hi[0])
        HI_circuit.iden(sub_q_hi[1])

        # Convert to a gate and stick it into an arbitrary place in the bigger circuit
        HI_instr = HI_circuit.to_instruction()
        circuit.append(HI_instr, [qr[0], qr[1]])

        circuit.cx(qr[0],qr[1])

        # Creates the second subcircuit that represents the gate H-H applied to
        # two qubits
        sub_q_hh = QuantumRegister(2)
        HH_circuit = QuantumCircuit(sub_q_hh, name='hh')
        HH_circuit.h(sub_q_hh[0])
        HH_circuit.h(sub_q_hh[1])

        HH_instr = HH_circuit.to_instruction()
        circuit.append(HH_instr, [qr[0], qr[1]])


        random_angles = [-1]
        while sum(random_angles) != 128:
            random_angles = list([randint(0, 127) for _ in range(nodes) ])

        #print('sum random angles', sum(random_angles))
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

        with open('res/'+DEBUG+'.txt','a') as f:
            #print('random angles:', sum(random_angles_steps), file = f)
            #print('fidelity:', fidelity, file = f)
            #print('counts:', counts_device, file = f)
            string = ''
            for key in counts_device:
                angles_dict[key] += 1
                string += (key+','+str(sum(random_angles_steps)))
                if xorBitByBit([int(elem) for elem in key]) == int(sum(random_angles_steps)/128)%2:
                    string += ',ok'
            print('{0:4.0f}'.format(fidelity*1000)+','+str(ITERATIONS)+'/'+str(iteration+1)+','+string, file = f)

    plot_histogram(angles_dict, title='Verification protocol results').savefig('res/'+DEBUG+'.png')

except QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))