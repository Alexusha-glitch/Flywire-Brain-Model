"""
FlyWire 3D Weighted Path Visualization
--------------------------------------
- Loads full connectivity + coordinate data
- Normalizes edge weights per source neuron
- Asks for a neuron ID
- Finds most likely 1000-step forward path (highest-weight traversal)
- Visualizes it in 3D using Plotly
"""

import re
import time
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from tqdm import tqdm

# --------------------------- Paths ---------------------------
connectivity_path = r"C:\Users\sysco\Downloads\FlyWire Raw Data\connections_princeton.csv\connections_princeton.csv"
coords_path       = r"C:\Users\sysco\Downloads\FlyWire Raw Data\coordinates.csv\coordinates.csv"

# --------------------------- Load Data ---------------------------
print("Loading connectivity CSV...")
connectivity = pd.read_csv(connectivity_path)
print(f"✅ Loaded {len(connectivity):,} edges")

print("Loading coordinates CSV...")
coords_df = pd.read_csv(coords_path)
print(f"✅ Loaded {len(coords_df):,} coordinate entries")

# --------------------------- Parse Coordinates (regex) ---------------------------
print("\nParsing [x y z] coordinates (regex-based, tolerant to any spacing/digits)...")

TRIPLE_INT_IN_BRACKETS = re.compile(r"\[?\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\s*\]?")
coords_dict = {}

for i, row in coords_df.iterrows():
    s = str(row.iloc[1])
    m = TRIPLE_INT_IN_BRACKETS.search(s)
    if not m:
        continue
    try:
        coords_dict[row.iloc[0]] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    except:
        continue

print(f"✅ Parsed {len(coords_dict):,} valid coordinates out of {len(coords_df):,}")

# --------------------------- Build Weighted Graph ---------------------------
print("\nBuilding directed weighted graph...")
G = nx.DiGraph()

for _, row in tqdm(connectivity.iterrows(), total=len(connectivity), desc="Adding edges"):
    try:
        src, dst, weight = str(row.iloc[0]), str(row.iloc[1]), float(row.iloc[3])
        if src not in G:
            G.add_node(src)
        if dst not in G:
            G.add_node(dst)
        G.add_edge(src, dst, raw_weight=weight)
    except Exception as e:
        continue

print(f"✅ Graph complete: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Normalize weights per source neuron
print("Normalizing edge weights per source neuron...")
for node in G.nodes():
    out_edges = list(G.out_edges(node, data=True))
    total = sum(e[2]['raw_weight'] for e in out_edges)
    if total > 0:
        for u, v, data in out_edges:
            data['weight'] = data['raw_weight'] / total
    else:
        for u, v, data in out_edges:
            data['weight'] = 0.0

print("✅ Weights normalized by source neuron total synapses.")

# --------------------------- Choose Start Neuron ---------------------------

manual_id = str(input("Enter neuron ID to activate (or leave blank for auto-selection): ")).strip()

if manual_id:
    start_id = manual_id
    if start_id not in G:
        raise ValueError(f"Neuron {start_id} not found in graph")
else:
    # Auto-select neuron with ~10 outgoing connections (chill but spreadable)
    target_degree = 10
    degree_dict = dict(G.out_degree())
    candidates = [n for n, d in degree_dict.items() if 8 <= d <= 15]

    if not candidates:
        candidates = [n for n, d in degree_dict.items() if 5 <= d <= 25]
    if not candidates:
        candidates = list(G.nodes)

    start_id = min(candidates, key=lambda n: abs(G.out_degree(n) - target_degree))
    print(f"⚙️ Auto-selected neuron {start_id} "
          f"with {G.out_degree(start_id)} outbound connections (target ≈10)")


# --------------------------- User Options ---------------------------

try:
    n_strongest = input("Enter number of strongest outgoing edges per neuron (blank = all): ").strip()
    n_strongest = int(n_strongest) if n_strongest else None
except:
    n_strongest = None

try:
    user_depth = input("Enter max propagation depth (blank = 100): ").strip()
    max_depth = int(user_depth) if user_depth else 100
except:
    max_depth = 100

max_nodes = 10000
decay_factor = 0.9  # how quickly activation fades

print(f"\nPropagating activation from {start_id} (max depth {max_depth}, "
      f"{'top ' + str(n_strongest) if n_strongest else 'all'} edges per neuron)...")

# ---------------------- BFS Propagation (Weighted Spread) ----------------------
visited = set([start_id])
frontier = [(start_id, 1.0)]
edges_to_plot, path_strong = [], []

for depth in tqdm(range(max_depth), desc="Depth levels"):
    next_frontier = []
    for neuron, act_strength in frontier:
        if neuron not in G:
            continue

        out_edges = list(G.out_edges(neuron, data=True))
        if not out_edges:
            continue

        # Sort edges by descending weight
        out_edges.sort(key=lambda e: e[2].get("weight", 0), reverse=True)

        # Keep top-n strongest if user specified
        if n_strongest:
            out_edges = out_edges[:n_strongest]

        # Track the strongest edge (for visual emphasis)
        path_strong.append(out_edges[0])

        for u, v, data in out_edges:
            w = data.get("weight", 0)
            if w <= 0:
                continue

            propagated_strength = act_strength * w * decay_factor
            edges_to_plot.append((u, v, propagated_strength))

            if v not in visited and len(visited) < max_nodes:
                visited.add(v)
                next_frontier.append((v, propagated_strength))

    frontier = next_frontier
    if not frontier or len(visited) >= max_nodes:
        break

print(f"✅ Spread reached {len(visited)} neurons, {len(edges_to_plot)} edges in {depth+1} depth levels")

# ---------------------- Diagnostics ----------------------
print("\n--- DEBUG INFO ---")
print(f"Sample visited neurons ({min(5, len(visited))}):", list(visited)[:5])
for n in list(visited)[:5]:
    print(f"Outgoing edges from {n}: {G.out_degree(n)}")

# ---------------------- Plotly 3D Visualization ----------------------
ids = np.array(list(coords_dict.keys()))
coords = np.array(list(coords_dict.values()))
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

# Background cloud
node_trace = go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(size=1.1, color="rgba(0,255,255,0.15)"),
    hoverinfo="none",
    name="All neurons"
)

# Activated neurons
# Convert both to strings for matching
visited = set(map(str, visited))
coords_dict = {str(k): v for k, v in coords_dict.items()}

# Now recompute active_coords
active_coords = np.array([coords_dict[n] for n in visited if n in coords_dict])

if active_coords.size == 0:
    print("⚠️ Warning: No matching coordinates found for activated neurons.")
    active_coords = np.zeros((1, 3))  # Prevent plotting error

active_trace = go.Scatter3d(
    x=active_coords[:, 0],
    y=active_coords[:, 1],
    z=active_coords[:, 2],
    mode="markers",
    marker=dict(size=2, color="rgb(0,180,255)", opacity=0.85),
    hoverinfo="none",
    name="Activated"
)



# Start neuron (large white)
if start_id in coords_dict:
    sx, sy, sz = coords_dict[start_id]
    start_trace = go.Scatter3d(
        x=[sx], y=[sy], z=[sz],
        mode="markers",
        marker=dict(size=7, color="white", opacity=1.0),
        name="Start"
    )
else:
    start_trace = []

# Weighted connection edges
# Weighted connection edges (color & width scale with propagated strength)
edge_traces = []

if edges_to_plot:
    max_strength = max(w for _, _, w in edges_to_plot if w > 0)
else:
    max_strength = 1.0

for u, v, w in edges_to_plot:
    if u not in coords_dict or v not in coords_dict:
        continue

    # Normalize strength (avoid division by zero)
    strength = max(0.0, min(1.0, w / max_strength))

    # Map strong edges to dark blue → weak edges to cyan
    # Higher strength = lower green intensity (darker color)
    g_value = int(255 - 200 * strength)   # 55 (dark blue) → 255 (light cyan)
    color_hex = f"rgb(0,{g_value},255)"

    x0, y0, z0 = coords_dict[u]
    x1, y1, z1 = coords_dict[v]

    edge_traces.append(go.Scatter3d(
        x=[x0, x1, None],
        y=[y0, y1, None],
        z=[z0, z1, None],
        mode="lines",
        line=dict(
            width=0.5 + 2.5 * strength,  # thicker for stronger edges
            color=color_hex
        ),
        opacity=0.85 * (0.3 + 0.7 * strength)  # more opaque for strong edges
    ))


# Strongest chain (dark blue highlight)
path_edges = []
for u, v, _ in path_strong:
    if u in coords_dict and v in coords_dict:
        x0, y0, z0 = coords_dict[u]
        x1, y1, z1 = coords_dict[v]
        path_edges.append(go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode="lines",
            line=dict(width=3, color="rgb(0,0,180)"),
            opacity=1.0
        ))

# Combine layers
fig = go.Figure(data=[node_trace, active_trace, start_trace] + edge_traces + path_edges)
fig.update_layout(
    title=f"FlyWire Activation Spread (depth {depth+1}, start {start_id})",
    showlegend=False,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor="black"
    ),
    paper_bgcolor="black",
    margin=dict(l=0, r=0, b=0, t=40)
)

output_path = f"FlywireGraph_Plotly3D_spread_{start_id}.html"
fig.write_html(output_path)
print(f"✅ Saved interactive 3D spread (depth {depth+1}) to {output_path}")