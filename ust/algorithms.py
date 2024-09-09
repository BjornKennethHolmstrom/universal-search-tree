def find_shortest_path(self, start_value, end_value):
    start = self.nodes[start_value]
    end = self.nodes[end_value]
    
    queue = [(start, [start_value])]
    visited = set()

    while queue:
        (node, path) = queue.pop(0)
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
