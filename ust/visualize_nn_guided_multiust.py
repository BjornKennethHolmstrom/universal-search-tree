import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nn_guided_adaptive_multi_ust import NeuralGuidedMultiUST
import numpy as np

def visualize_ust(ust, title="Neural Network-Guided AdaptiveMultiUST"):
    G = nx.Graph()
    pos_3d = {}
    for i, node in enumerate(ust.nodes):
        G.add_node(i)
        pos_3d[i] = node.coords
        for dim in range(ust.dimensions):
            for connected_node in node.connections[dim]:
                connected_index = ust.nodes.index(connected_node)
                G.add_edge(i, connected_index)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw nodes
    xs = [coord[0] for coord in pos_3d.values()]
    ys = [coord[1] for coord in pos_3d.values()]
    zs = [coord[2] for coord in pos_3d.values()]
    ax.scatter(xs, ys, zs, c='lightblue', s=500)
    
    # Draw edges
    for edge in G.edges():
        x = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
        y = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
        z = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
        ax.plot(x, y, z, c='gray')
    
    # Draw labels
    for i, (x, y, z) in pos_3d.items():
        ax.text(x, y, z, f'{i}', fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('nn_guided_multiust_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create and populate a sample UST
ust = NeuralGuidedMultiUST(dimensions=3)
for _ in range(20):
    coords = np.random.rand(3)
    ust.add_node(coords)

# Visualize the initial UST
visualize_ust(ust, title="Initial Neural Network-Guided AdaptiveMultiUST")

# Perform some operations to potentially trigger restructuring
for _ in range(100):
    query_coords = np.random.rand(3)
    ust.find_nearest(query_coords)

# Visualize the UST after operations
visualize_ust(ust, title="Neural Network-Guided AdaptiveMultiUST After Operations")
