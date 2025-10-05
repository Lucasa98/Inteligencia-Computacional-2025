import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from ACO import ACO

def cargarCSV(csvRelativePath, delimeter = ',') -> np.ndarray[any]:
    this_folder = os.path.abspath('') 
    fullPath = os.path.join(this_folder, csvRelativePath)
    return np.loadtxt(fullPath, dtype=float, delimiter=delimeter)

def plot_aco_path(G_matrix, path, node_positions=None, pheromones=None, title=None):
    n = G_matrix.shape[0]
    G = nx.Graph()

    # Add edges with distances
    for i in range(n):
        for j in range(i+1, n):
            if G_matrix[i, j] > 0:
                G.add_edge(i, j, weight=G_matrix[i, j])

    if node_positions is None:
        node_positions = nx.spring_layout(G, seed=42)

    # Edge colors (pheromone intensity or default)
    if pheromones is not None:
        pheromone_vals = [pheromones[i, j] for i, j in G.edges()]
        edge_colors = pheromone_vals
        cmap = plt.cm.plasma
    else:
        edge_colors = "gray"
        cmap = None

    # Draw base graph
    nx.draw(
        G, pos=node_positions, with_labels=True,
        node_color="lightblue", node_size=600,
        edge_color=edge_colors, width=2,
        font_weight="bold", font_size=10,
        cmap=cmap
    )

    # Highlight best path
    if path:
        path_edges = [(i, j) for (i, j) in path]
        nx.draw_networkx_edges(
            G, pos=node_positions, edgelist=path_edges,
            edge_color="red", width=3
        )

        total_distance = sum(G_matrix[i, j] for (i, j) in path)
        path_nodes = [path[0][0]] + [b for (_, b) in path]
        plt.title(title or f"Best path {path_nodes} — Distance: {total_distance:.2f}")

    plt.axis("off")
    plt.show()

data = cargarCSV(os.path.join('Práctica','Guia7','data','gr17.csv'))

colonia = ACO(cant_hormigas=20,
              tasa_evaporacion=0.5,
              feromonas_depositadas=1,
              metodo='uniforme',
              alpha=4,
              beta=4
)

path = colonia.buscar(data, 0, 8)
print(path)
plot_aco_path(data, path, pheromones=colonia.σ)