import numpy as np

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

