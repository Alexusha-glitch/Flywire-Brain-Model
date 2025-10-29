import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from pyvis.network import Network
from tqdm import tqdm
import time

# --------------------------- Custom Class ---------------------------
class neurone:
    def __init__(self, connections, activation, id):
        self.connections = connections
        self.activation = activation
        self.id = id

    def activate(self):
        totalSynapses = np.sum(self.connections[:, 1])
        self.connections[:, 0].activation = self.activation * self.connections[:, 1] / totalSynapses

# --------------------------- Load Data ---------------------------
print("Loading connectivity CSV...")
connectivity = pd.read_csv(
    r"C:\Users\sysco\Downloads\FlyWire Raw Data\connections_princeton.csv\connections_princeton.csv"
)
print(f"Loaded {len(connectivity):,} edges")

# --------------------------- Build NetworkX Graph ---------------------------
print("\nBuilding NetworkX graph...")

# tqdm progress bar on edge creation
G = nx.DiGraph()
for _, row in tqdm(connectivity.iterrows(), total=len(connectivity), desc="Adding edges"):
    G.add_edge(row.iloc[0], row.iloc[1], weight=row.iloc[3])

print(f"✅ Graph complete: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# --------------------------- Build PyVis Graph ---------------------------
print("\nConverting to PyVis network...")

net = Network(height="900px", width="100%", bgcolor="#111", font_color="white", directed=True)

# Add nodes with progress bar
for node in tqdm(G.nodes(), total=G.number_of_nodes(), desc="Adding nodes"):
    net.add_node(str(node), title=f"Neuron {node}")

# Add edges with progress bar
for u, v, data in tqdm(G.edges(data=True), total=G.number_of_edges(), desc="Adding edges to PyVis"):
    net.add_edge(str(u), str(v), value=data.get("weight", 1))

print("✅ Conversion complete. Exporting to HTML...")

try:
    # try normal notebook rendering first
    net.show("FlywireGraph.html", notebook=True)
except Exception as e:
    print(f"⚠️ PyVis show() failed: {e}")
    print("Falling back to write_html()...")
    net.write_html("FlywireGraph.html")
    print("✅ Graph HTML written directly to FlywireGraph.html")

# --------------------------- Export ---------------------------