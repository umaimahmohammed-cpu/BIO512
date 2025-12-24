#final code for project 2
# BIO512 - Project 2: Gene Co-expression Network Analysis
import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 1) Load network
coexp_network = pd.read_csv("coexpression_network.txt", sep="\t")
coexp_network = coexp_network.dropna(subset=["gene1", "gene2", "weight"]) # Remove rows with missing gene names or weights

# Create iGraph graph
edges = list(zip(coexp_network["gene1"], coexp_network["gene2"], coexp_network["weight"])) # 
g = ig.Graph.TupleList(edges, directed=False, edge_attrs=["weight"]) # Load edges with weights

# Basic graph info
print("Vertices:", g.vcount())
print("Edges:", g.ecount())
print("Density:", g.density())
print("diameter ", g.diameter()) 


# 2) Convert weights to distances for shortest-path based centralities
eps = 1e-9 
g.es["dist"] = [1.0 / (w + eps) for w in g.es["weight"]] #

# 3) Centrality (weighted where appropriate)
degree_c = g.strength(weights="weight")  # weighted degree (more biologically meaningful than raw degree)
betweenness_c = g.betweenness(weights="dist")
closeness_c = g.closeness(weights="dist", normalized=True)
eigenvector_c = g.eigenvector_centrality(weights="weight")

# Store as attributes
g.vs["strength"] = degree_c
g.vs["betweenness"] = betweenness_c
g.vs["closeness"] = closeness_c
g.vs["eigenvector"] = eigenvector_c

centrality_df = pd.DataFrame({
    "gene": g.vs["name"],
    "strength": degree_c,
    "betweenness": betweenness_c,
    "closeness": closeness_c,
    "eigenvector": eigenvector_c
})

#  Identify top 10 genes by each centrality metric
deg_s = pd.Series(degree_c, index=g.vs["name"]).sort_values(ascending=False) 
bet_s = pd.Series(betweenness_c, index=g.vs["name"]).sort_values(ascending=False)
clo_s = pd.Series(closeness_c, index=g.vs["name"]).sort_values(ascending=False)
eig_s = pd.Series(eigenvector_c, index=g.vs["name"]).sort_values(ascending=False)

print("\nTop 10 genes by Degree Centrality:\n", deg_s.head(10))
print("\nTop 10 genes by Betweenness Centrality:\n", bet_s.head(10))
print("\nTop 10 genes by Closeness Centrality:\n", clo_s.head(10))
print("\nTop 10 genes by Eigenvector Centrality:\n", eig_s.head(10))

# 4) Visualization (size=eigenvector, color=betweenness)
eig = np.array(eigenvector_c) 
bet = np.array(betweenness_c)

# Robust scaling
eig_scaled = (eig - eig.min()) / (eig.max() - eig.min() + 1e-9) 
bet_scaled = (bet - bet.min()) / (bet.max() - bet.min() + 1e-9)

node_sizes = 10 + 40 * eig_scaled  # keep sizes reasonable
norm = Normalize(vmin=bet_scaled.min(), vmax=bet_scaled.max())
node_colors = [plt.cm.plasma(norm(v)) for v in bet_scaled]

# Edge widths based on weight strength
w = np.array(g.es["weight"])
w_scaled = (w - w.min()) / (w.max() - w.min() + 1e-9)
edge_widths = 0.2 + 2.0 * w_scaled

layout = g.layout("fr")

fig, ax = plt.subplots(figsize=(10, 8))
ig.plot(
    g,
    target=ax,
    layout=layout,
    vertex_size=node_sizes,
    vertex_color=node_colors,
    vertex_label=None,
    edge_width=edge_widths,
    edge_color="gray",
)
plt.title("Neu-M5 Centrality-Weighted Co-expression Network\n(size=eigenvector, color=betweenness)")
plt.show()
