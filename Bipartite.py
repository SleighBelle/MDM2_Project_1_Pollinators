import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# File Path
df = pd.read_csv("strengths.csv")

# Column Name (Could be Replaced)
Col_A = "pollinator_name"
Col_B = "robustness_score"

# Two Columns be Plotted
required_cols = {Col_A, Col_B}
if not required_cols.issubset(df.columns):
    raise ValueError(f"No Such Column: {required_cols}") # Check

# Bip Graph
B = nx.Graph()

# Left: Col_A
# Right：Col_B
for _, row in df.iterrows():
    pollinator = row[Col_A]
    score = row[Col_B]
# Adding Nodes:
    score_node = f"R={score}"
    B.add_node(pollinator, bipartite=0)   # left
    B.add_node(score_node, bipartite=1)   # right
# Adding Edge
    B.add_edge(pollinator, score_node)

# Plotting
left_nodes  = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
right_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]

pos = nx.bipartite_layout(B, left_nodes)

plt.figure(figsize=(10, 5))
nx.draw_networkx_nodes(B, pos, nodelist=left_nodes, node_size=800)
nx.draw_networkx_nodes(B, pos, nodelist=right_nodes, node_shape='s', node_size=800)
nx.draw_networkx_edges(B, pos)
nx.draw_networkx_labels(B, pos, font_size=8)

# Andrew Plot update
plt.rcParams.update({
    'font.size': 16,        # general text
    'axes.titlesize': 18,   # title
    'axes.labelsize': 16,   # x/y labels
    'xtick.labelsize': 14,  # x tick labels
    'ytick.labelsize': 14,  # y tick labels
    'legend.fontsize': 14   # legend text (if you add one later)
    })


plt.axis("off")
plt.title(f"Bipartite Graph of {Col_A} and {Col_B}")
plt.tight_layout()
plt.show()
