import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

input_path = 'MDM2/outputs/strengths.csv'
df = pd.read_csv(input_path)

# Column names
col_a = "pollinator_name"   
col_b = "plant_name"        
weight_col = "robustness_score"
constituents = [col_a, col_b, weight_col]

# Check these columns exist
columns = df.columns 
for constituent in constituents:
    if not constituent in columns:
        raise ValueError(f'❌ Column {constituent} not in the provided dataframe')

#building the graph
G = nx.Graph()

# Partition sets
pollinators = df[col_a].unique()
plants = df[col_b].unique()

# Add nodes with bipartite attribute
G.add_nodes_from(plants, bipartite = "plants")
G.add_nodes_from(pollinators, bipartite = "pollinators")

# Add weighted edges between plants (i) and pollinators (j)
for index, row in df.iterrows():
    pollinator = row[col_a]
    plant = row[col_b]
    weight = float(row[weight_col])

    G.add_edge(plant, pollinator, weight=weight)

# Position nodes in bipartite layout (plants on one side, pollinators on the other)
pos = nx.bipartite_layout(G, plants)

# Extract edge weights and scale them for line width
edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
if edge_weights:
    max_w = max(edge_weights)
    # Scale to a reasonable width range
    widths = [0.5 + 4.5 * (w / max_w) for w in edge_weights]
else:
    pass
    #widths = 1.0

plt.figure(figsize=(16, 10))

# Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=plants, node_color="lightgreen", label="Plants")
nx.draw_networkx_nodes(G, pos, nodelist=pollinators, node_color="lightblue", label="Pollinators")

# Draw weighted edges
nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7)

# Labels
nx.draw_networkx_labels(G, pos, font_size=8)

plt.legend()
plt.title("Plant-Pollinator Bipartite Network (edge weight = robustness_score)")
plt.axis("off")
plt.tight_layout()
plt.show()
