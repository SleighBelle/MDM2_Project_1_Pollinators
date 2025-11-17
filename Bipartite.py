import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# File Input
df = pd.read_csv("strengths.csv")

# Column Name 
Col_A = "pollinator_name"
Col_B = "robustness_score"   # Weight Side 

# Column Checking 
required_cols = {Col_A, Col_B}
if not required_cols.issubset(df.columns):
    raise ValueError(f"No Such Column: {required_cols}")
# Andrew Plot update
plt.rcParams.update({
    'font.size': 16,        # general text
    'axes.titlesize': 18,   # title
    'axes.labelsize': 16,   # x/y labels
    'xtick.labelsize': 14,  # x tick labels
    'ytick.labelsize': 14,  # y tick labels
    'legend.fontsize': 14   # legend text
})

# Weighted Bipartite Graph 
B = nx.Graph()

for _, row in df.iterrows():
    Start = row[Col_A]
    score = row[Col_B]

    # Right Node Name 
    score_node = f"R={score}"

    # Nodes on Two Sides
    B.add_node(Start, bipartite=0)   # left
    B.add_node(score_node, bipartite=1)   # right

    # Adding weight on Edges
    B.add_edge(Start, score_node, weight=float(score))

# Dividing Nodes by bipartite 
left_nodes  = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
right_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]

pos = nx.bipartite_layout(B, left_nodes)

# Plotting 
plt.figure(figsize=(20, 10))

nx.draw_networkx_nodes(B, pos, nodelist=left_nodes, node_size=800)
nx.draw_networkx_nodes(B, pos, nodelist=right_nodes, node_shape='s', node_size=800)

nx.draw_networkx_edges(B, pos)

nx.draw_networkx_labels(B, pos, font_size=8)

# Edge Label (Weight Number)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in B.edges(data=True)}
nx.draw_networkx_edge_labels(B, pos, edge_labels=edge_labels, font_size=8)

plt.axis("off")
plt.title(f"Weighted Bipartite Graph of {Col_A} and {Col_B}")
plt.tight_layout()
plt.show()

# Bar Chart
plt.figure(figsize=(20, 10))

x = range(len(df))

plt.bar(x, df[Col_B])
plt.xticks(x, df[Col_A], rotation=90)
plt.ylabel(Col_B)
plt.xlabel(Col_A)
plt.title(f"{Col_B} for each {Col_A}")
plt.tight_layout()
plt.show()
