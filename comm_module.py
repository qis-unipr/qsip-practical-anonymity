import json
import platform
import socket

import simulaqron
import os
CONFIG_PATH = os.path.dirname(simulaqron.__file__) + "/config/network.json"
BROADCAST_PORT = 12345
LISTENING_PORT = 8000


def createBroadcastServer():
    # creates the broadcast socket
    broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    # creates the listening socket that receives the message to broadcast
    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listening_sock.bind(('localhost', LISTENING_PORT))

    while True:
        msg_to_broadcast = listening_sock.recv(2000).decode('utf-8')
        print('server received: '+ str(msg_to_broadcast))
        broadcast_sock.sendto(bytes(msg_to_broadcast, 'utf-8'), ('<broadcast>', BROADCAST_PORT))


# Accesses to SimulaQron network options and retrieves the port of the node
def getConfigPort(id=None):
    if id is not None:
        with open(CONFIG_PATH, 'r') as config_file:
            data = json.load(config_file)
        return data['default']['nodes'][id]['app_socket'][1]
    return None



# This class creates a separate thread that handles the classical communications
class CommunicationManager():
    def __init__(self, id):
        self._id = str(id)
        self._messages = []

        self._broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if platform.system() != 'Windows':
            self._broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._broadcast_sock.bind(('', BROADCAST_PORT))

        self._listening_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if platform.system() != 'Windows':
            self._broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._listening_sock.bind(('localhost', int(getConfigPort(self._id)+1000)))

    def sendBroadcastMessage(self, msg):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        msg = msg+','+self._id
        sock.sendto(bytes(msg, 'utf-8'), ('localhost', LISTENING_PORT))     

    def receiveBroadcastMessage(self):
        while True:
            data, addr = self._broadcast_sock.recv(2000).decode('utf-8').split(',')

            if addr != self._id:
                self._messages.append((data, addr))

    def sendMessageToNode(self, msg, id):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(bytes(str(msg), 'utf-8'), ('localhost', getConfigPort(id)+1000))

    def sendMessageToNodeWithId(self, msg, id):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(bytes(str(msg)+','+str(self._id)[-1], 'utf-8'), ('localhost', getConfigPort(id)+1000))
        
    def recvMessage(self):
        msg, sender = self._listening_sock.recvfrom(2000)
        return msg.decode('utf-8'), sender


    def bufferRead(self):
        A = self._messages[:]
        self._messages = []
        return A