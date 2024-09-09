# ust/utils.py

from .core import UniversalSearchTree, USTNode
from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt

def visualize_tree(ust: UniversalSearchTree, filename: str = 'ust_visualization.png'):
    """
    Visualize the Universal Search Tree using networkx and matplotlib.
    
    Args:
    ust (UniversalSearchTree): The UST to visualize.
    filename (str): The filename to save the visualization. Default is 'ust_visualization.png'.
    """
    G = nx.DiGraph()

    def add_edges(node: USTNode):
        for child in node.children:
            G.add_edge(node.value, child.value)
            add_edges(child)

    # Add all nodes and edges to the graph
    for node in ust.nodes.values():
        G.add_node(node.value)
        add_edges(node)

    # Create the visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, arrows=True)
    
    # Add labels to edges
    edge_labels = {(u, v): '' for (u, v) in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Universal Search Tree Visualization")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(filename)
    plt.close()

    print(f"Tree visualization saved as {filename}")

# You can add more utility functions here as needed
