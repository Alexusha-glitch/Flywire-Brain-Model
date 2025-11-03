import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
from tqdm import tqdm
import time

# --------------------------- Load Data ---------------------------
print("Loading connectivity CSV...")
connectivity = pd.read_csv(
    r"C:\Users\sysco\Downloads\FlyWire Raw Data\connections_princeton.csv\connections_princeton.csv"
)
print(f"✅ Loaded {len(connectivity):,} edges")

# --------------------------- Build Graph ---------------------------
print("\nBuilding NetworkX directed graph...")

G = nx.DiGraph()
for _, row in tqdm(connectivity.iterrows(), total=len(connectivity), desc="Adding edges"):
    # assumes: source, target, ..., weight in 4th column
    try:
        src, dst, weight = row.iloc[0], row.iloc[1], row.iloc[3]
        G.add_edge(src, dst, weight=weight)
    except Exception:
        continue

print(f"✅ Graph complete: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# --------------------------- Convert to PyVis ---------------------------
print("\nConverting to PyVis network (browser-optimized)...")

net = Network(
    height="1000px",
    width="100%",
    bgcolor="#0d0d0d",
    font_color="white",
    directed=True,
    notebook=False,
)

# --- super important: disable physics explosions for large graphs ---
# Uses hierarchical layout for faster rendering, no lag
net.set_options("""
var options = {
  nodes: {
    shape: 'dot',
    size: 3,
    font: { size: 8, color: '#ffffff' }
  },
  edges: {
    color: { inherit: 'both' },
    width: 0.1,
    smooth: false
  },
  physics: {
    enabled: true,
    solver: 'barnesHut',
    barnesHut: {
      gravitationalConstant: -3000,
      springLength: 45,
      springConstant: 0.005,
      damping: 0.9,
      avoidOverlap: 0.3
    },
    stabilization: { iterations: 50, fit: true }
  },
  interaction: {
    hover: true,
    tooltipDelay: 100,
    zoomView: true,
    navigationButtons: true,
    keyboard: true
  },
  layout: {
    improvedLayout: false
  }
}
""")

# --- add nodes in one sweep (no tqdm here to avoid slow console I/O) ---
for node in G.nodes():
    net.add_node(str(node), title=f"Neuron {node}")

# --- batch edge addition (chunked to speed up memory use) ---
edges = list(G.edges(data=True))
chunk_size = 50000
for i in range(0, len(edges), chunk_size):
    batch = edges[i:i+chunk_size]
    for u, v, data in batch:
        net.add_edge(str(u), str(v), value=data.get("weight", 1))
    print(f"  Added edges {i:,}–{i+len(batch):,} / {len(edges):,}")

print("✅ Conversion complete. Writing to FlywireGraph.html ...")

# --------------------------- Export ---------------------------
t0 = time.time()
net.write_html("FlywireGraph.html")
print(f"✅ Done! HTML saved as FlywireGraph.html ({time.time()-t0:.2f}s)")
print("Open it manually in Chrome or Edge (double-click the file).")
print("If it loads slowly, wait a few seconds — all nodes/edges are included.")
