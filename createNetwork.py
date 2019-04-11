import multiprocessing
from simulaqron.network import Network
from sys import argv
import subprocess
import os

if __name__ == '__main__':
    n_nodes = int(argv[1])
    tmp = [ 'node'+str(i) for i in range(0, n_nodes) ]
    print("nodes: ",tmp)

    network = Network(nodes=tmp, topology='complete')
    network.start()

    for i in range(0, n_nodes):
        params = "python3 node.py "+str(n_nodes)+" "+str(i)+" "
        
        if ( i == n_nodes-1):
            params += "1 "
        else:
            params += "0 "
        #if i != n_nodes-1:
        params += "&"

        subprocess.call(params, shell=True)