# ust/adaptive_multi_ust.py

import numpy as np
from collections import defaultdict

class MultiDimNode:
    def __init__(self, coords, data=None):
        self.coords = np.array(coords)
        self.data = data
        self.connections = defaultdict(list)  # dim -> list of connected nodes
        self.access_count = 0

    def connect(self, other_node, dimension):
        self.connections[dimension].append(other_node)
        other_node.connections[dimension].append(self)

    def disconnect(self, other_node, dimension):
        self.connections[dimension].remove(other_node)
        other_node.connections[dimension].remove(self)

class AdaptiveMultiUST:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.nodes = []
        self.access_threshold = 10  # Threshold for restructuring

    def add_node(self, coords, data=None):
        node = MultiDimNode(coords, data)
        self.nodes.append(node)
        self._connect_to_nearest(node)
        return node

    def _connect_to_nearest(self, node):
        for dim in range(self.dimensions):
            nearest = min(self.nodes, key=lambda n: abs(n.coords[dim] - node.coords[dim]) if n != node else float('inf'))
            if nearest != node:
                node.connect(nearest, dim)

    def find_nearest(self, coords):
        target = np.array(coords)
        current = self.nodes[0] if self.nodes else None
        path = []

        while current:
            path.append(current)
            current.access_count += 1
            if current.access_count > self.access_threshold:
                self._restructure(current)

            best_neighbor = None
            best_distance = np.linalg.norm(current.coords - target)

            for dim, neighbors in current.connections.items():
                for neighbor in neighbors:
                    dist = np.linalg.norm(neighbor.coords - target)
                    if dist < best_distance:
                        best_neighbor = neighbor
                        best_distance = dist

            if best_neighbor is None:
                break
            current = best_neighbor

        return path[-1], path

    def _restructure(self, node):
        # Simple restructuring: disconnect from least accessed neighbors
        for dim, neighbors in node.connections.items():
            least_accessed = min(neighbors, key=lambda n: n.access_count)
            node.disconnect(least_accessed, dim)
            self._connect_to_nearest(least_accessed)
        node.access_count = 0

# Example usage
ust = AdaptiveMultiUST(dimensions=3)
ust.add_node([1, 2, 3], "A")
ust.add_node([4, 5, 6], "B")
ust.add_node([7, 8, 9], "C")

nearest, path = ust.find_nearest([3, 4, 5])
print(f"Nearest node to [3, 4, 5]: {nearest.data}")
print(f"Path: {[node.data for node in path]}")
