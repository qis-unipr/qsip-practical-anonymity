# utilities 
from ast import literal_eval
from bitstring import BitArray, Bits
import json
from math import log, ceil, acos, pi
import numpy as np
from random import randint, uniform, shuffle, choice
from time import sleep, time

# SimulaQron module
from cqc.pythonLib import CQCConnection, qubit

# OS modules
from sys import argv
import os

# Other modules
from comm_module import CommunicationManager
from createNetwork import DEBUG
from ast import literal_eval

############################################################################
################################  CONSTANTS  ###############################
N_ITERATIONS = 300
PI_VALUE = 128
TOLERANCE = 0.1  # this value is needed in order to exit in the GHZStateGenerator's while loop
TWO_PI_STEPS = 256
SLEEPTIME = 0.05

lvl_verb = 1

############################################################################
######################  COMMUNICATION FUNCTIONS  ###########################
# MESSAGES
MSG_SKIP = 101
MSG_OK_VERIFICATION = 102
MSG_ABORT_PROTOCOL = 103
MSG_REPEAT_PROTOCOL = 104

# 0: classic
# 1: ket 0^n distribuito
strategy = 0
meas = None


# This function broadcasts a message in a network.
# @param order: the order that must be followed in the broadcasting
# @param msg: the message that should be broadcasted
# @param myval: if True appends to the results the msg
# @return the function returns a list of received values
def broadcastSingleValue(cm, node_id, order, msg, myVal=False, skipcount=0):
    received_vals = []

    for _ in range(0, order.index(node_id)):
        data = int(cm.recvMessage()[0])
        if data != MSG_SKIP:
            received_vals.append(data)

    if myVal and msg != MSG_SKIP:
        received_vals.append(msg)

    for node in order:
        if node != node_id:
            cm.sendMessageToNode(int(msg), 'node' + str(node))

    for _ in range(order.index(node_id), len(order) - 1):
        data = int(cm.recvMessage()[0])
        if data != MSG_SKIP:
            received_vals.append(data)

    sleep(SLEEPTIME)
    return received_vals


# This function lets a node to broadcast a BitArray list in a network.
# @param order: the order that must be followed in the broadcasting
# @param values: this is the list of values that the node wants to broadcast
def broadcastBitArray(cm, node_id, order, values):
    received_vals = BitArray()
    for _ in range(order.index(node_id)):
        data = int(cm.recvMessage()[0])
        if data != MSG_SKIP:
            data = Bits(bin=str(data))
            received_vals.append(data)

    received_vals.append(Bits(bin=values.bin[order.index(node_id)]))
    for i, node in enumerate(order):
        if node != node_id:
            cm.sendMessageToNode(int(values[i]), 'node' + str(node))

    for _ in range(order.index(node_id), len(order) - 1):
        data = int(cm.recvMessage()[0])
        if data != MSG_SKIP:
            data = Bits(bin=str(data))
            received_vals.append(data)

    sleep(SLEEPTIME)
    return received_vals


# This function lets a node to broadcast a list in a network.
# Sends the i-th element to the i-th agent
# @param nodes: the order that must be followed in the broadcasting
# @param msg: this is the list of values that the node wants to broadcast
def broadcastOrderedValues(conn, node_id, nodes, msg=None):
    if msg is None:
        data = int(list(conn.recvClassical())[0])
    else:
        for i, node in enumerate(nodes):
            if node != node_id:
                # data = 0
                sendMessage(conn, 'node' + str(node), [msg[i]])
            else:
                data = msg[i]
    sleep(SLEEPTIME)
    return int(data)


def sendMessage(conn, dest, msg):
    send_success = False
    while not send_success:
        try:
            conn.sendClassical(dest, msg)
            send_success = True
        except:
            conn.closeClassicalChannel(dest)


############################################################################
##############################  PROTOCOLS  #################################
# Adding noise to qubit. The value depends on ε value given in the createNetwork.py script. 
# The angle is calculated in steps relying on the following formula: ((number_of_steps/2)*ε)/n_nodes
def ApplyNoise(q_list, qubits, rots):
    for idx, q in enumerate(qubits):
        if rots[idx] == 0:  # rotation on y axis
            q_list[q].rot_Y(1)
        else:  # rotation on x axis
            q_list[q].rot_X(1)
    return q_list


# The goal of the Anonymous Entanglement Protocol is the creation a shared EPR
# pair between Sender and Receiver. This function returns the EPR generated
def AnonymousEntanglementProtocol(conn, cm, node_id, q, order, sender=False, receiver=False):
    buffer = []
    msg = MSG_SKIP
    if not receiver and not sender:
        q.H()
        msg = q.measure()
        printOnConsole(node_id, "broadcasting measure: " + str(msg))
    buffer.extend(broadcastSingleValue(cm, node_id, order, msg, True))

    msg = MSG_SKIP
    if sender:
        msg = int(randint(0, 1))
        if msg == 1:
            q.Z()
        printOnConsole(node_id, 'broadcasting random bit: ' + str(msg))
    buffer.extend(broadcastSingleValue(cm, node_id, order, msg, True))

    msg = MSG_SKIP
    if receiver:
        msg = int(randint(0, 1))

        if int(xorBitByBit(BitArray(buffer)).bin) == 1:
            q.Z()
        printOnConsole(node_id,
                       'random bit: ' + str(msg) + " buffer: " + str(buffer) + " xor: " + xorBitByBit(
                           BitArray(buffer)).bin)
        broadcastSingleValue(cm, node_id, order, msg, True)
    else:
        buffer.extend(broadcastSingleValue(cm, node_id, order, msg))

    printOnConsole(node_id, 'Received values: ' + str(buffer))


# The source generates a state |Ψ⟩ and distributes it to the agents.
# In this case the source is the last  It generates GHZ state and sends a qubit to each agent
# The quantum circuit used is based on:
# https://dal.objectstorage.open.softlayer.com/v1/AUTH_42263efc45184c7ca4742512588a1942/codes/code-5cabae1dd7559f0053f4e4b6.png
def GHZStateGenerator(cm, conn, node_id, order, fidelity=1, adversary=False, adversaries=None):
    global meas
    n_nodes = len(order)

    if node_id == n_nodes - 1:
        printOnConsole(node_id, 'distributing GHZ states...', 1)
        q = qubit(conn)

        remaining_nodes = order[:]
        remaining_nodes.remove(node_id)
        qubit_list = []

        # applies noise for the qubits
        if fidelity < 1:
            with open("output" + str(n_nodes) + "_" + str(int(fidelity * 100)) + ".txt",
                      "r") as rotations_file:
                lines = rotations_file.readlines()

            line = randint(0, len(lines) - 1)
            rotations = lines[line].split(';')[1:][0]
            rotations = literal_eval(rotations)
            qubits, rotations = [list(tup) for tup in zip(*rotations)]

        if strategy == 0:
            for _ in range(len(remaining_nodes)):
                q_send = qubit(conn)
                q_send.H()
                q_send.cnot(q)
                qubit_list.append(q_send)

            for qbit in qubit_list:
                qbit.H()
            q.H()
        else:
            q.H()
            for node in remaining_nodes:
                q_send = qubit(conn)
                q_send.H()
                qubit_list.append(q_send)
        qubit_list.append(q)

        if fidelity < 1:
            qubit_list = ApplyNoise(qubit_list, qubits, rotations)

        q = qubit_list[-1]

        for node, q_to_send in enumerate(qubit_list[:-1]):
            conn.sendQubit(q_to_send, 'node' + str(node))
    else:
        q = conn.recvQubit()

    if adversary and strategy == 0:
        q.H()
        meas = q.measure(inplace=True)

    return q


# Notification protocol
# The Sender notifies the Receiver: agents run the Notification protocol.
def NotificationProtocol(cm, S, order, node_id, sender=False):
    n_nodes = len(order)
    x_list = [0 for i in range(n_nodes)]

    if sender:
        sender_position = order.index(node_id)
        receiver = sender_position

        # A random receiver is extracted by the sender
        while receiver == sender_position:
            receiver = randint(0, n_nodes - 1)
        x_list[receiver] = 1
        printOnConsole(node_id, 'Notification protocol. Chosen receiver: node' + str(order[receiver]))

    receiver = False
    for pos, elem in enumerate(order):
        if elem == node_id:
            printOnConsole(node_id, "waiting for notification...")
        y_list = BitArray()

        for _ in range(S):
            p_j = Bits(bin=str(x_list[pos]))

            if x_list[pos] == 1 and elem != node_id:
                p_j = Bits(bin=str(randint(0, 1)))
            seq = paritySequence(p_j, n_nodes)

            r_list = BitArray(broadcastBitArray(cm, node_id, order, seq))
            z_j = xorBitByBit(r_list)

            # the i-th agent doesn't broadcast his value
            if elem == node_id:
                msg = MSG_SKIP
                z_list = BitArray(broadcastSingleValue(cm, node_id, order, msg))
                z_list.append(z_j)
                y_list.append(xorBitByBit(z_list))
            else:
                msg = int(z_j.bin)
                broadcastSingleValue(cm, node_id, order, msg)

        if elem == node_id:
            for i in y_list:
                receiver = receiver or i
            receiver = Bits(bool=receiver)
    return receiver


# Random Agent Protocol.
# It is an extension of RandomBit protocol. It must be performed log_2(n) times
# and the result is the ID of the Verifier. Every node knows who is the Verifier
# Added the possibility of forcing honest verifiers
def RandomAgentProtocol(cm, S, order, node_id, sender=False):
    n_nodes = len(order)

    network_nodes = [x for x in range(n_nodes)]
    verifier = BitArray()

    while len(verifier) == 0 or not (verifier.uint in network_nodes):
        cheater = False
        verifier = BitArray()

        for _ in range(int(ceil(log(n_nodes, 2)))):
            if sender:
                x_i = Bits(bin=str(bin(round(np.random.uniform(0, 1)))))
            else:
                x_i = Bits(bin='0')

            y_i = str(RandomBitProtocol(cm, S, order, node_id, x_i, sender))

            if sender:
                to_print = ""
                if y_i != x_i.bin:
                    to_print += "Someone cheated in RandomBitProtocol"
                    cheater = True
                else:
                    to_print += "No one cheated in RandomBitProtocol"
                to_print = to_print + " y_i: " + y_i + " x_i: " + x_i.bin
                printOnConsole(node_id, to_print, 1)

            verifier.append(Bits(bin=y_i))
            if verifier.uint in network_nodes and sender:
                printOnConsole(node_id, 'Extracted a node that is not in the network, repeating protocol', 1)

    # forces the verifier to be an honest one
    # only for simulation purpose
    if sender:
        try:
            with open('adv.json', "r") as adv_file:
                adv = json.load(adv_file)['adversary']
                adv = list(map(int, adv))

            if int(verifier.uint) in adv:
                printOnConsole(node_id, 'Adversary found, repeating protocol', 1)
                return -2
        except:
            pass

    if sender and cheater:
        return -1
    return int(verifier.uint)


# Random Bit Protocol
# The sender's goal is choosing a bit according to a distribution.
# It is based on LogicalOR protocol (and Parity)
def RandomBitProtocol(cm, S, order, node_id, x_i, sender=False):
    n_nodes = len(order)

    y_i = 0
    if sender:
        print('\n')
        printOnConsole(node_id, "Executing RandomBit protocol")

    for actual_iteration in range(n_nodes):
        if sender:
            printOnConsole(node_id, "iteration: " + str(actual_iteration + 1))

        y_list = BitArray()
        for _ in range(S):

            # choose p_i
            if int(x_i.bin) == 1:
                p_i = Bits(bin=str(randint(0, 1)))
            else:
                p_i = Bits(bin='0')

            # PARITY PROTOCOL
            # Step 1: generate sequence
            seq = paritySequence(p_i, n_nodes)
            # printOnConsole("random bits chosen: "+ seq.bin +"(x_i: "+ x_i.bin + ", p_i chosen: "+ p_i.bin +")")

            # Step 2: send jth bit to jth agent
            r_list = broadcastBitArray(cm, node_id, order, seq)

            # Steps 3-4: compute z_j and broadcast it to other nodes
            z_j = xorBitByBit(r_list)
            z_list = BitArray(broadcastSingleValue(cm, node_id, order, int(z_j.bin), True))

            y_list.append(xorBitByBit(z_list))
            # if sender:
            #    printOnConsole(node_id,"computed y_i: "+ str(xorBitByBit(z_list)) +" DEBUG: z sequence: "+ z_list.bin)

        if y_i == 0:
            for bit in y_list:
                if int(bit) == 1:
                    y_i = 1
                    break
        order = shiftListLeftByOne(order)

    sleep(SLEEPTIME)
    return y_i


# Verification Protocol
# This protocol is a GHZ verification of |Ψ⟩ for the agents.
# The Verifier would like to verify how close this shared state is to the ideal state 
# and whether or not it contains GME.
def VerificationProtocol(conn, cm, q, order, node_id, verifier=None, adversary=False):
    n_nodes = len(order)

    if verifier == node_id:
        printOnConsole(node_id, 'I\'m the verifier. Running Verification protocol')
        random_angles = [-1]

        # Extracts random angles. The sum must be a multiple of pi. 
        # Steps are 2*PI/256 radians, so PI = 128 and 2PI is 256
        while sum(random_angles) % PI_VALUE != 0:
            random_angles = list([randint(0, 127) for _ in range(n_nodes)])

        printOnConsole(node_id, "Sum random angles: " + str(sum(random_angles)) + " " + str(random_angles))
        angle = broadcastOrderedValues(conn, node_id, order, random_angles)
    else:
        angle = broadcastOrderedValues(conn, node_id, order)

    if angle != 0:
        q.rot_Z(TWO_PI_STEPS - angle)
    q.rot_Y(TWO_PI_STEPS - 64)
    measure = q.measure()

    if verifier == node_id:
        results = [measure]
        user = [node_id]
        while len(results) < n_nodes:
            received_message = cm.recvMessage()[0]
            received_message = received_message.split(',')
            results.append(int(received_message[0]))
            user.append(int(received_message[1]))

        tuprint = sorted(list(zip(user, results)), key=lambda tup: tup[0])
        _, tuprint = [list(tup) for tup in zip(*tuprint)]
        printOnConsole(node_id, "results: " + str(results), 0)
    else:
        if adversary:
            if strategy == 0:
                measure = meas

            if angle >= int(PI_VALUE / 2):
                measure = (measure + 1) % 2

        cm.sendMessageToNodeWithId(measure, 'node' + str(verifier))
        sleep(SLEEPTIME * n_nodes)

    msg = MSG_SKIP
    if verifier == node_id:
        # calculates if the sum of every angle is odd or even
        odd_pi_mul = int(sum(random_angles) / PI_VALUE) % 2
        y_j = xorBitByBit(BitArray(results))
        printOnConsole(node_id,
                       "multiple of pi (1 odd, 0 even): " + str(odd_pi_mul) + " y_j calculated: " + str(
                           int(y_j.bin)))

        if int(y_j.bin) == odd_pi_mul:
            printOnConsole(node_id, 'No one is cheating.')
            msg = MSG_OK_VERIFICATION
        else:
            printOnConsole(node_id, 'Someone is cheating.')
            msg = MSG_ABORT_PROTOCOL

        to_append = ""

        if adversary:
            printOnConsole(node_id, 'The adversary is the verifier, cheating on response')
            to_append += ",cheater_as_verifier_(" + str(msg) + ")"
            msg = MSG_OK_VERIFICATION

        writeout("_simulation", n_nodes, 'verification protocol,' +
                 str(N_ITERATIONS) + "/" + str(iteration + 1) + "," + str(msg) +
                 ",measures: " + str(tuprint) +
                 ",angles: " + str(random_angles) + " " + str(sum(random_angles)) + to_append)
    else:
        sleep(SLEEPTIME * n_nodes)

    return measure, angle, broadcastSingleValue(cm, node_id, order, msg, True)[0]


############################################################################
##########################  UTILITY FUNCTIONS  #############################
def intToBinaryString(n_nodes, number):
    result = Bits(bin=str(bin(number))).bin
    if len(result) < int(ceil(log(n_nodes, 2))):  # n_nodes.bit_length():
        prefix = ''.join(['0' for _ in range(
            int(log(n_nodes, 2)) - len(result))])  # range(n_nodes.bit_length()-len(result))])
        result = prefix + result
    return result


def xorBitByBit(bits):
    if len(bits) == 0:
        return -1

    result = bits[0]
    for idx in range(1, len(bits)):
        result = result ^ bits[idx]
    return Bits(bin=bin(result))


# This function generates a sequence that xored bit-by-bit matches the x_i in input
# @param x_i: the bit to match with the sequence
# @param n: length of the sequence
def paritySequence(x_i, n):
    condition = False
    r_list = BitArray()
    while not condition:
        r_list = BitArray()
        for _ in range(0, n):
            r_list.append(Bits(bin=str(randint(0, 1))))

        if xorBitByBit(r_list).bin == x_i.bin:
            condition = True
    return r_list


def printOnConsole(node_id, message, verbosity=0):
    if verbosity <= lvl_verb:
        print('ID:', 'node' + str(node_id), '--->', message)


def shiftListLeftByOne(input):
    tmp = [input[-1]]
    tmp.extend(input[:-1])
    return tmp


def writeout(protocol, n_nodes, result):
    with open("results/" + DEBUG + protocol + "_" + str(n_nodes) + ".csv", "a") as output:
        print(result, file=output)


############################################################################
####################################  MAIN  ################################
def main():
    global iteration
    try:
        with open('./conf.json', 'r') as conf:
            data = json.load(conf)
            n_nodes = data['params']['n_nodes']
            fidelity = data['params']['fidelity']
            order = list(data['ordering'])
            S = data['params']['S']
            lvl_verb = data['params']['verbose']
    except:
        print('conf.json file file not found!')
        exit()

    if not os.path.exists("output" + str(n_nodes) + "_{0:2.0f}.txt".format(fidelity * 100)) and fidelity < 1:
        print('Missing rotations file')
        exit()

    node_id = int(argv[1])
    sender = False
    adversary = False
    adversaries = None

    if argv[2] == '1':
        sender = True
    elif argv[2] == '2':
        adversary = True
        adversaries = list(
            map(int, list(argv[3].split('-'))))  # all the dishonest agents know the other adversaries
        printOnConsole(node_id, 'allies: ' + str(adversaries), 2)
    printOnConsole(node_id, 'conf.json loaded!', 2)

    communication_manager = CommunicationManager('node' + str(node_id))
    printOnConsole(node_id, 'Communication manager created!', 2)
    sleep(SLEEPTIME)

    with CQCConnection('node' + str(node_id)) as conn:
        for iteration in range(N_ITERATIONS):
            # The Sender notifies the Receiver
            receiver = NotificationProtocol(communication_manager, S, order, node_id, sender)
            if int(receiver.bin) == 1:
                printOnConsole(node_id, 'I\'m the receiver')

            # GHZ state generation
            q = GHZStateGenerator(communication_manager, conn, node_id, order, fidelity, adversary,
                                  adversaries)

            if node_id == n_nodes - 1:
                printOnConsole(node_id, 'GHZ distributed\n')

            # The Sender anonymously chooses Verification or Anonymous Entanglement
            # The agents perform the RandomBit protocol, with the Sender choosing her input according to the 
            # following probability distribution: she flips S fair classical coins, and if all coins are heads, s
            # he inputs 0, else she inputs 1. Let the outcome be x.
            # 0 → head, 1 → cross
            coins = []
            if sender:
                print('\n')
                printOnConsole(node_id, 'flipping coins', 1)
                for _ in range(S):
                    coins.append(randint(0, 1))
                coins = sum(coins)
                if coins == 0:
                    printOnConsole(node_id, "RandomBit = 0, all heads")
                else:
                    coins = 1
                    printOnConsole(node_id, "RandomBit = 1")
                x_i = Bits(bin=str(coins))
            else:
                x_i = Bits(bin='0')

            y_i = RandomBitProtocol(communication_manager, S, order, node_id, x_i, sender)

            msg = MSG_SKIP
            if sender:
                to_print = ""
                if y_i != int(x_i.bin):
                    to_print += "Someone cheated in RandomBitProtocol"
                    writeout("_simulation", n_nodes, str(N_ITERATIONS) + "/" + str(iteration + 1) + str(
                        MSG_ABORT_PROTOCOL) + ", RandomBitProtocol")
                    msg = MSG_ABORT_PROTOCOL
                else:
                    to_print += "No one cheated in RandomBitProtocol"
                    msg = MSG_OK_VERIFICATION

                if lvl_verb == 2:
                    to_print = to_print + " y_i: " + str(y_i) + " x_i: " + x_i.bin
                printOnConsole(node_id, to_print)
            else:
                printOnConsole(node_id, "y_i: " + str(y_i), 2)

            # in order to simulate the protocol abortation, we use this broadcast. Every process
            # skips an iteration if randombit fails. In reality the protocol should continue
            # and the sender pretends to be a normal agent 
            protocol_ok = broadcastSingleValue(communication_manager, node_id, order, msg)
            if protocol_ok == MSG_ABORT_PROTOCOL or msg == MSG_ABORT_PROTOCOL:
                try:
                    q.release()
                except:
                    printOnConsole(node_id, 'qubit already released')
                continue
            sleep(SLEEPTIME)

            if y_i == 0:
                printOnConsole(node_id, 'Running Anonymous Entanglement')
                AnonymousEntanglementProtocol(conn, communication_manager, node_id, q, order, sender,
                                              receiver)
            else:
                printOnConsole(node_id, 'Running RandomAgent protocol')

                # The following cycle allows to select always an honest verifier if the variable is set. 
                # This is useful when we want to simulate a specific situation of the protocol   
                repeat = True
                next_iteration = False
                while repeat:
                    verifier = RandomAgentProtocol(communication_manager, S, order, node_id, sender)

                    msg = MSG_SKIP
                    if sender:
                        if verifier == -1:
                            printOnConsole(node_id, 'Someone cheated in RandomAgentProtocol')
                            writeout("_simulation", node_id,
                                     str(N_ITERATIONS) + "/" + str(iteration + 1) + str(
                                         msg) + ", RandomAgentProtocol")
                            protocol_ok = MSG_ABORT_PROTOCOL
                        elif verifier == -2:
                            printOnConsole(node_id, "Extracting another verifier")
                            protocol_ok = MSG_REPEAT_PROTOCOL
                        else:
                            printOnConsole(node_id, 'No one cheated in RandomAgentProtocol')
                            protocol_ok = MSG_OK_VERIFICATION
                        broadcastSingleValue(communication_manager, node_id, order, protocol_ok)
                    else:
                        protocol_ok = broadcastSingleValue(communication_manager, node_id, order, msg)[0]
                    repeat = False
                    if protocol_ok == MSG_ABORT_PROTOCOL:
                        try:
                            q.release()
                        except:
                            printOnConsole(node_id, 'Qubit already released')
                        next_iteration = True
                    elif protocol_ok == MSG_REPEAT_PROTOCOL:
                        repeat = True
                    sleep(SLEEPTIME)
                if next_iteration:
                    continue

                ############################################################################################
                printOnConsole(node_id, 'node' + str(verifier) + ' is the verifier')
                VerificationProtocol(conn, communication_manager, q, order, node_id, verifier, adversary)
        if sender:
            writeout("_simulation", n_nodes, '\n')


if __name__ == "__main__":
    main()
