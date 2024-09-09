import numpy as np
from sklearn.neural_network import MLPRegressor
from adaptive_multi_ust import AdaptiveMultiUST, MultiDimNode

class NeuralGuidedMultiUST(AdaptiveMultiUST):
    def __init__(self, dimensions):
        super().__init__(dimensions)
        self.nn_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000)
        self.training_data = []

    def _get_node_features(self, node):
        return np.concatenate([
            node.coords,
            [len(node.connections[dim]) for dim in range(self.dimensions)],
            [node.access_count]
        ])

    def _restructure(self, node):
        features = self._get_node_features(node)
        self.training_data.append((features, [len(node.connections[dim]) for dim in range(self.dimensions)]))
        
        if len(self.training_data) > 100:  # Train only when we have enough data
            X = np.array([data[0] for data in self.training_data])
            y = np.array([data[1] for data in self.training_data])
            self.nn_model.fit(X, y)
            
            predicted_connections = self.nn_model.predict([features])[0]

            # Use predictions to guide restructuring
            for dim in range(self.dimensions):
                current_connections = set(node.connections[dim])
                desired_connections = set(sorted(self.nodes, 
                    key=lambda n: abs(n.coords[dim] - node.coords[dim]))[:int(predicted_connections[dim])])
                
                # Add new connections
                for new_node in desired_connections - current_connections:
                    node.connect(new_node, dim)
                
                # Remove unnecessary connections
                for old_node in current_connections - desired_connections:
                    node.disconnect(old_node, dim)

        node.access_count = 0

# Example usage
ust = NeuralGuidedMultiUST(dimensions=3)
for _ in range(100):
    coords = np.random.rand(3)
    ust.add_node(coords)

for _ in range(1000):
    query_coords = np.random.rand(3)
    nearest, path = ust.find_nearest(query_coords)
    print(f"Query: {query_coords}, Nearest: {nearest.coords}, Path length: {len(path)}")
