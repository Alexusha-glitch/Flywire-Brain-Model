import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network

# Steps to continue activation
# Find connections to other neuron

class neurone:
    def __init__(self, connections, activation, id):
        self.connections = connections
        self.activation = activation
        self.id = id

    def activate(self):
        totalSynapses = np.sum(self.connections[:, 1])
        self.connections[:, 0].activation = self.activation * self.connections[:, 1] / totalSynapses

connectivity = pd.read_csv("C:\\Users\\sysco\\Downloads\\FlyWire Raw Data\\connections_princeton.csv\\connections_princeton.csv")
print(connectivity)

# TRY IMPLEMENTING CLASS; IMPLEMENT COORDINATES INTO DIGRAPH

brain = nx.digraph()
for i in range(len(connectivity)):
    brain.add_edge(connectivity[i][0], connectivity[i][1], weight=connectivity[i][3])

