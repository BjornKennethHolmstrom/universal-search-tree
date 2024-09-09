from collections import deque

class USTNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parents = []
        self.is_leaf = True

class UniversalSearchTree:
    def __init__(self):
        self.nodes = {}

    def add_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = USTNode(value)
        return self.nodes[value]

    def connect_nodes(self, parent_value, child_value):
        parent = self.add_node(parent_value)
        child = self.add_node(child_value)
        parent.children.append(child)
        child.parents.append(parent)
        parent.is_leaf = False

    def cut_tree(self, node_value):
        node = self.nodes[node_value]
        for parent in node.parents:
            parent.children.remove(node)
        node.parents = []

    def merge_trees(self, root1_value, root2_value):
        root1 = self.nodes[root1_value]
        root2 = self.nodes[root2_value]
        root1.children.append(root2)
        root2.parents.append(root1)
        root1.is_leaf = False

    def find_shortest_path(self, start_value, end_value):
        if start_value not in self.nodes or end_value not in self.nodes:
            return None

        start = self.nodes[start_value]
        end = self.nodes[end_value]
        
        queue = deque([(start, [start_value])])
        visited = set()

        while queue:
            node, path = queue.popleft()
            if node not in visited:
                visited.add(node)

                if node == end:
                    return path

                for child in node.children:
                    if child not in visited:
                        queue.append((child, path + [child.value]))
                for parent in node.parents:
                    if parent not in visited:
                        queue.append((parent, path + [parent.value]))

        return None  # No path found

