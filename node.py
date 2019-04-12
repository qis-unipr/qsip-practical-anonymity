from bitstring import BitArray, Bits
from cqc.pythonLib import CQCConnection, qubit
from math import log
from simulaqron.network import Network
from sys import argv
import numpy as np
from random import randint
from time import sleep

SLEEPTIME = 0.3
TWO_PI_STEPS = 255

########################################################################################################################################
############################################################  MESSAGES  ################################################################
MSG_HELLO = 100
MSG_SKIP = 101
MSG_OK_VERIFICATION = 102
MSG_ABORT_PROTOCOL = 103
########################################################################################################################################

arguments = 4
n_nodes = -1
node_addr = ''
node_id = -1
sender = False

def printOnConsole(message):
    print('ID:', node_addr ,'--->',message)

def listOfNodes(nodes):
    nodeList = []
    for idx in range(0, nodes):
        idx_name = 'node'+str(idx)
        if( idx_name != node_addr ):
            nodeList.append(idx_name)
    return nodeList

def xorBitByBit(bits):
    if( len(bits) == 0):
        return -1

    result = bits[0] 
    for idx in range(1, len(bits)):
        result = result^bits[idx]
    return Bits( bin=bin(result) )

# This function generates a sequence that xored bit-by-bit matches the x_i in input
# @param x_i: the bit to match with the sequence
# @param n: length of the sequence
def paritySequence(x_i, n):
    condition = False
    r_list = BitArray()
    while( not condition ):
        r_list = BitArray()
        for _ in range(0, n):
            r_list.append( Bits(bin=str(randint(0,1))) )
        
        if( xorBitByBit(r_list).bin == x_i.bin ):
            condition = True
    return r_list

def shiftListLeftByOne(input):
    tmp = [input[-1]]
    tmp.extend(input[:-1])
    return tmp

############################################################################
######################  COMMUNICATION FUNCTIONS  ###########################
def sendMessage(dest, msg):
    send_success = False
    while(not send_success):
        try:
            conn.sendClassical(dest, msg)
            send_success = True
        except:
            conn.closeClassicalChannel(dest)

# This function broadcasts a message in a network.
# @param order: the order that must be followed in the broadcasting
# @param msg: the message that should be broadcasted
# @param myval: if True appends to the results the msg
# @return the function returns a list of received values
def broadcastSingleValue(order, msg, myVal=False):
    received_vals = []

    for _ in range(0, order.index(node_id)):
        data = list(conn.recvClassical())[0]
        if data != MSG_SKIP:
            received_vals.append(data)
    
    if myVal and msg != MSG_SKIP:
        received_vals.append(msg)

    for node in order:
        if(node != node_id):
            sendMessage('node'+str(node), [msg])
        
    for _ in range(order.index(node_id), len(order)-1):
        data = list(conn.recvClassical())[0]
        if data != MSG_SKIP:
            received_vals.append(data)
    sleep(SLEEPTIME)
    return received_vals

# This function lets a node to broadcast a BitArray list in a network.
# @param order: the order that must be followed in the broadcasting
# @param values: this is the list of values that the node wants to broadcast
def broadcastBitArray(order, values):
    received_vals = BitArray()

    for _ in range(order.index(node_id)):
        data = list(conn.recvClassical())[0]
        if data != MSG_SKIP:
            data = Bits(bin=str(data))
            received_vals.append(data)
    
    received_vals.append(Bits(bin=values.bin[order.index(node_id)]))
    for i,node in enumerate(order):
        if(node != node_id):
            sendMessage( 'node'+str(node), [int(values[i])] )
        
    for _ in range(order.index(node_id), len(order)-1 ):
        data = list(conn.recvClassical())[0]
        if data != MSG_SKIP:
            data = Bits(bin=str(data))
            received_vals.append(data)
    sleep(SLEEPTIME)
    return received_vals

# This function lets a node to broadcast a list in a network.
# Sends the i-th element to the i-th agent
# @param nodes: the order that must be followed in the broadcasting
# @param msg: this is the list of values that the node wants to broadcast
def broadcastOrderedValues(nodes, msg):
    if msg == None:
        data = list(conn.recvClassical())[0]
    else:
        for i,node in enumerate(nodes):
            if node != node_id:
                sendMessage( 'node'+str(node), [msg[i]] )
            else:
                data = msg[i]
    return int(data)

########################################################################################################################################
########################################################  PROTOCOLS  ###################################################################

# The goal of the Anonymous Entanglement Protocol  is creating a shared EPR 
# pair between Sender and Receiver. This function returns the EPR generated
def AnonymousEntanglementProtocol(q):
    buffer = []
    msg = MSG_SKIP
    if not receiver and not sender:
        q.H()
        msg = q.measure()
        printOnConsole("broadcasting measure: " + str(msg))
    buffer.extend(broadcastSingleValue(order, msg, True))

    msg = MSG_SKIP
    if sender:
        if cheater:
            q.H()
            msg = q.measure()
            printOnConsole("Acting like normal agent. Broadcasting measure: " + str(msg))
        else:
            msg = int(randint(0,1))
            if msg == 1: 
                q.Z()
            printOnConsole('broadcasting random bit: '+ str(msg))
    buffer.extend(broadcastSingleValue(order, msg, True))

    msg = MSG_SKIP
    if receiver:
        msg = int(randint(0,1))

        if int(xorBitByBit(BitArray(buffer)).bin) == 1:
            q.Z()
        printOnConsole('random bit: '+ str(msg) + " buffer: "+ str(buffer)+ " xor: "+xorBitByBit(BitArray(buffer)).bin)
        broadcastSingleValue(order, msg, True) 
    else:
        buffer.extend(broadcastSingleValue(order, msg))

    printOnConsole('Received values: ' + str(buffer))
    
    if receiver or sender:
        printOnConsole("DEBUG: q measure: "+ str(q.measure(inplace = True)))
        return q
    return None

# The source generates a state |Ψ⟩ and distributes it to the agents.
# In this case the source is the last node. It generates GHZ state and sends a qubit to each agent
# The quantum circuit used is based on:
# https://dal.objectstorage.open.softlayer.com/v1/AUTH_42263efc45184c7ca4742512588a1942/codes/code-5cabae1dd7559f0053f4e4b6.png
def GHZStateGenerator():  
    if node_addr == 'node'+str(n_nodes-1):
        print('\n')
        printOnConsole('distributing GHZ states...')
        q = qubit(conn)
        
        remaining_nodes = order.copy()
        remaining_nodes.remove(node_id)

        to_send =[]
        for node in remaining_nodes:
            q_send = qubit(conn)
            q_send.H()
            q_send.cnot(q)
            to_send.append(q_send)

        for idx, node in enumerate(remaining_nodes):
            to_send[idx].H()
            conn.sendQubit(to_send[idx], 'node'+str(node))
        q.H()

    else:
        q = conn.recvQubit()
    return q

# Notification protocol
# The Sender notifies the Receiver: agents run the Notification protocol.
def NotificationProtocol(order, agent_to_notify=None):
    x_list = [ 0 for i in range(n_nodes) ]
    if agent_to_notify != None:
        x_list[agent_to_notify] = 1

    receiver = False
    for pos, elem in enumerate(order):
        if elem == node_id:
            printOnConsole("waiting for notification...")
        y_list = BitArray()

        for _ in range(S):
            p_j = Bits(bin=str(x_list[pos]))
            
            if x_list[pos] == 1 and elem != node_id:
                p_j = Bits(bin=str(randint(0,1)))
            seq = paritySequence(p_j, n_nodes)

            r_list = BitArray(broadcastBitArray(order, seq))
            z_j = xorBitByBit(r_list)  

            #the i-th agent doesn't broadcast his value
            if elem == node_id:
                msg = MSG_SKIP
                z_list = BitArray( broadcastSingleValue(order, msg) )
                z_list.append(z_j)
                y_list.append(xorBitByBit(z_list))
            else:
                msg = int(z_j.bin)
                broadcastSingleValue(order, msg)

        if elem == node_id:
            for i in y_list:
                receiver = receiver or i
            receiver = Bits(bool=receiver)
    return receiver

# Random Agent Protocol.
# It is an extension of RandomBit protocol. It must be performed log_2(n) times
# and the result is the ID of the Verifier. Every node knows who is the Verifier
def RandomAgentProtocol(order):
    verifier = BitArray()
    for _ in range(int(log(n_nodes,2))):
        if sender:
            x_i = Bits(bin=str(bin(round(np.random.uniform(0,1)))))
            printOnConsole('random bit: '+x_i.bin)
        else:
            x_i = Bits(bin='0')
            
        y_i = str(RandomBitProtocol(x_i, order))
        
        if sender:
            if y_i != x_i.bin:
                printOnConsole("Someone is cheating. y_i: "+ y_i +" x_i: "+x_i.bin)
                #inserire gestione dei cheater
                #cheater = True
            else:
                printOnConsole("All fine. y_i: "+ y_i+" x_i: "+x_i.bin)
        verifier.append( Bits(bin=y_i) )
    return int(verifier.uint)

# Random Bit Protocol
# The sender's goal is choosing a bit according to a distribution.
# It is based on LogicalOR protocol (and Parity)
def RandomBitProtocol(x_i, order):
    y_i = 0
    if sender:
        print('\n')
        printOnConsole("Executing RandomBit protocol")
    
    for iteration in range(n_nodes):
        if sender:
            printOnConsole("iteration: "+ str(iteration+1))
        y_list = BitArray()
        for _ in range(S):
            #choose p_i
            if int(x_i.bin) == 1:
                p_i = Bits(bin=str(randint(0,1)))
            else:
                p_i = Bits(bin='0')

            #PARITY PROTOCOL
            #Step 1: generate sequence
            seq = paritySequence(p_i, n_nodes)
            #printOnConsole( "random bits chosen: "+ seq.bin +"(x_i: "+ x_i.bin + ", p_i chosen: "+ p_i.bin +")")
            
            #Step 2: send jth bit to jth agent
            r_list = broadcastBitArray(order, seq)
            
            #Steps 3-4: compute z_j and broadcast it to other nodes
            z_j = xorBitByBit(r_list) 
            z_list = BitArray(broadcastSingleValue(order, int(z_j.bin), True))

            y_list.append(xorBitByBit(z_list))
            #if sender:
            #    printOnConsole("computed y_i: "+ str(xorBitByBit(z_list)) +" DEBUG: z sequence: "+ z_list.bin)

        if y_i == 0:
            for bit in y_list:
                if int(bit) == 1:
                    y_i = 1
                    break

        order = shiftListLeftByOne(order) 
    return y_i

# Verification Protocol
# This protocol is a GHZ verification of |Ψ⟩ for the agents.
# The Verifier would like to verify how close this shared state is to the ideal state 
# and whether or not it contains GME.
def VerificationProtocol(order, verifier):
    if verifier == node_id:          
        print('\n')
        printOnConsole('I\'m the verifier. Running Verification protocol')
        random_angles = [-1]

        #extract random angles. the sum must be multiple of pi. Steps are 2*PI/256 radians
        while sum(random_angles)%128 != 0:
            random_angles = list([ randint(0, 127) for _ in range(n_nodes) ])
        printOnConsole("sum random_angles: "+ str(sum(random_angles))+" "+str(random_angles))
        angle = broadcastOrderedValues(order, random_angles)
    else:
        angle = broadcastOrderedValues(order, None)

    qubit.rot_Z(TWO_PI_STEPS - angle)
    qubit.rot_Y(TWO_PI_STEPS - 63)

    measure = qubit.measure()
    results = [measure]

    if verifier == node_id:
        msg = MSG_SKIP
    else:
        msg = measure

    results.extend(broadcastSingleValue(order, msg))
    printOnConsole("DEBUG: measure: "+str(measure)+ " buffer "+ str(results)+ " angle: "+str(angle))

    if verifier == node_id:
        #is the sum of every angle odd or even?
        odd_pi_mul = int(sum(random_angles)/128)%2
        y_j = xorBitByBit(BitArray(results))
        printOnConsole( "multiple of pi (1 odd, 0 even): "+ str(odd_pi_mul) +" y_j calculated: " + str(int(y_j.bin)))

        if int( y_j.bin ) == odd_pi_mul:
            printOnConsole('No one is cheating.')
            msg = MSG_OK_VERIFICATION
        else:
            printOnConsole('Someone is cheating')
            msg = MSG_ABORT_PROTOCOL
    else:
        msg = MSG_SKIP
    
    return broadcastSingleValue(order, msg, True)[0]

########################################################################################################################################
##########################################################  MAIN  ######################################################################
if __name__ == '__main__':
    if len(argv) < arguments :
        print('missing arguments')
        exit()

    n_nodes = int( argv[1] )
    node_addr = 'node'+argv[2]
    node_id = int(node_addr[-1])
    sender = bool(int(argv[3]))

    if len(argv) > arguments:
        fetuso = True
        printOnConsole('I\'m the fetuso')

    node_list = listOfNodes(n_nodes)
    
    order = [ i for i in range(n_nodes)] #[ i for i in range(n_nodes-1,-1,-1) ]#
    if sender:
        cheater = False
        printOnConsole('Ordering: '+str(order))

    printOnConsole('checking connection...')
    msg = MSG_SKIP
    S = 10

    with CQCConnection(node_addr) as conn:
        if sender:
            printOnConsole( 'sending hello message to others...')
            msg = MSG_HELLO

        messages = broadcastSingleValue(order, msg)
        if len(messages) > 0:
            printOnConsole('Received '+ str(messages[0]))


        # A random receiver is extracted by the sender
        receiver = None
        if sender:
            receiver = order.index(node_id)
            while receiver == order.index(node_id):
                receiver = randint(0, n_nodes-1)
            print('\n')
            printOnConsole('Notification protocol. Chosen receiver: node'+str(order[receiver]))
        receiver = NotificationProtocol(order, receiver)

        #printOnConsole("DEBUG: resulting y_i: " + receiver.bin)
        if receiver:
            printOnConsole('I\'m the receiver')
        
        # GHZ state generation
        # TODO: Inserire la generazione degli stati da una sorgente non affidabile (?)
        qubit = GHZStateGenerator()

        if node_addr == 'node'+str(order[-1]):
            printOnConsole('GHZ state created!')
        #printOnConsole("DEBUG: GHZ " + str(q.measure(inplace=True)))

        # The Sender anonymously chooses Verification or Anonymous Entanglement
        # The agents perform the RandomBit protocol, with the Sender choosing her input according to the 
        # following probability distribution: she flips S fair classical coins, and if all coins are heads, s
        # he inputs 0, else she inputs 1. Let the outcome be x.
        # 0 → head, 1 → cross
        coins = []
        if sender:
            print('\n')
            printOnConsole('flipping coins')
            for i in range(S):
                coins.append(randint(0,1))
            coins = sum(coins)
            if coins == 0:
                printOnConsole("RandomBit = 0, all heads")
            else:
                coins = 1
                printOnConsole("RandomBit = 1")
            x_i = Bits(bin=str(coins))
        else:
            x_i = Bits(bin='0')

        y_i = RandomBitProtocol(x_i, order)

        if sender:
            if y_i != int(x_i.bin):
                printOnConsole("Someone is cheating. y_i: "+ str(y_i)+" x_i: "+x_i.bin)
                cheater = True
            else:
                printOnConsole("All fine. y_i: "+ str(y_i)+" x_i: "+x_i.bin)
        else:
            printOnConsole("y_i: "+ str(y_i))
        sleep(SLEEPTIME)

        if sender:
            print('\n\n')
        if y_i == 0:
            # Anonymous Entanglement
            printOnConsole('Running Anonymous Entanglement')   
            qubut = AnonymousEntanglementProtocol(qubit)
        else:
            # Random agent protocol
            printOnConsole('Running RandomAgent protocol')

            verifier = RandomAgentProtocol(order)
            printOnConsole('Node' + str(verifier) +' is the verifier')

            # Verification protocol
            verification_result = VerificationProtocol(order, verifier)

            if verification_result == MSG_OK_VERIFICATION:
                printOnConsole('Verification ok')
            elif verification_result == MSG_ABORT_PROTOCOL:
                printOnConsole('Someone tried to cheat, aborting protocol')
