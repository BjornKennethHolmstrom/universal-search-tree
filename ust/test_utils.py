# ust/test_utils.py

from ust.core import UniversalSearchTree
import random

def generate_random_ust(num_nodes: int, max_children: int) -> UniversalSearchTree:
    """
    Generate a random Universal Search Tree for testing purposes.

    Args:
    num_nodes (int): The number of nodes to generate.
    max_children (int): The maximum number of children for each node.

    Returns:
    UniversalSearchTree: A randomly generated UST.
    """
    ust = UniversalSearchTree()
    nodes = [f'Node_{i}' for i in range(num_nodes)]
    
    # Add all nodes to the tree
    for node in nodes:
        ust.add_node(node)
    
    # Randomly connect nodes
    for i, node in enumerate(nodes):
        num_children = random.randint(0, min(max_children, num_nodes - i - 1))
        children = random.sample(nodes[i+1:], num_children)
        for child in children:
            ust.connect_nodes(node, child)
    
    return ust

def verify_ust_properties(ust: UniversalSearchTree) -> bool:
    """
    Verify that the given UST satisfies basic properties.

    Args:
    ust (UniversalSearchTree): The UST to verify.

    Returns:
    bool: True if the UST satisfies all checked properties, False otherwise.
    """
    # Check that all nodes in the tree are unique
    if len(ust.nodes) != len(set(node.value for node in ust.nodes.values())):
        print("Error: Duplicate nodes found in the tree.")
        return False

    # Check that all parent-child relationships are consistent
    for node in ust.nodes.values():
        for child in node.children:
            if node not in child.parents:
                print(f"Error: Inconsistent parent-child relationship between {node.value} and {child.value}")
                return False

    # Check that leaf nodes are correctly marked
    for node in ust.nodes.values():
        if node.is_leaf != (len(node.children) == 0):
            print(f"Error: Leaf status is incorrect for node {node.value}")
            return False

    return True

def compare_paths(path1: list, path2: list) -> bool:
    """
    Compare two paths to check if they are equivalent.

    Args:
    path1 (list): The first path.
    path2 (list): The second path.

    Returns:
    bool: True if the paths are equivalent, False otherwise.
    """
    return len(path1) == len(path2) and all(a == b for a, b in zip(path1, path2))

# You can add more test utility functions here as needed
