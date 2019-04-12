import multiprocessing
from simulaqron.network import Network
from sys import argv
import subprocess
import os

N_NODES = 4

if __name__ == '__main__':
    if len(argv) > 1:
        N_NODES = int(argv[1])

    node_list = [ 'node'+str(i) for i in range(0, N_NODES) ]
    print("Creating Network with nodes: ", node_list, "\n")

    network = Network(nodes=node_list, topology='complete')
    network.start()

    for i in range(0, N_NODES):
        # invocation params: python3 node.py <number of nodes in network> <node_name> <sender> <adversary> &
        params = "python3 node.py "+str(N_NODES)+" "+str(i)+" "
        
        if ( i == N_NODES-2):
            params += "1 "
        else:
            params += "0 "

        params += "&"

        subprocess.call(params, shell=True)