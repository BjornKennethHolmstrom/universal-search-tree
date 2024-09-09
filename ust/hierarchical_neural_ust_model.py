import numpy as np
import tensorflow as tf
from adaptive_multi_ust import AdaptiveMultiUST

class HierarchicalNeuralUST:
    def __init__(self, input_dim, output_dim, ust_dimensions, num_subnets):
        self.ust = AdaptiveMultiUST(dimensions=ust_dimensions)
        self.subnets = []
        self.optimizers = []
        
        for _ in range(num_subnets):
            subnet = self._create_subnet(input_dim, output_dim)
            optimizer = tf.keras.optimizers.Adam(0.01)
            self.subnets.append(subnet)
            self.optimizers.append(optimizer)
            
            coords = np.random.rand(ust_dimensions)
            self.ust.add_node(coords, len(self.subnets) - 1)

    def _create_subnet(self, input_dim, output_dim):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(output_dim)
        ])

    def call(self, inputs):
        nearest, _ = self.ust.find_nearest(np.random.rand(self.ust.dimensions))
        active_subnet = self.subnets[nearest.data]
        return active_subnet(inputs)

    @tf.function
    def compute_loss(self, subnet, inputs, targets):
        predictions = subnet(inputs)
        return tf.reduce_mean(tf.square(predictions - targets))

    @tf.function
    def compute_gradients(self, subnet, inputs, targets):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(subnet, inputs, targets)
        return tape.gradient(loss, subnet.trainable_variables), loss

    def train_step(self, inputs, targets):
        nearest, _ = self.ust.find_nearest(np.random.rand(self.ust.dimensions))
        active_subnet = self.subnets[nearest.data]
        active_optimizer = self.optimizers[nearest.data]

        gradients, loss = self.compute_gradients(active_subnet, inputs, targets)
        active_optimizer.apply_gradients(zip(gradients, active_subnet.trainable_variables))

        return loss

# Example usage
input_dim = 10
output_dim = 1
ust_dimensions = 3
num_subnets = 5
model = HierarchicalNeuralUST(input_dim, output_dim, ust_dimensions, num_subnets)

# Generate some sample data
X = np.random.normal(size=(1000, input_dim)).astype(np.float32)
y = np.sum(X, axis=1, keepdims=True).astype(np.float32)

# Train the model
for epoch in range(100):
    epoch_loss = 0
    for i in range(0, len(X), 32):
        batch_X = X[i:i+32]
        batch_y = y[i:i+32]
        epoch_loss += model.train_step(batch_X, batch_y)
    print(f"Epoch {epoch}, Loss: {epoch_loss/len(X)}")

# Evaluate the model
test_X = np.random.normal(size=(100, input_dim)).astype(np.float32)
test_y = np.sum(test_X, axis=1, keepdims=True).astype(np.float32)
predictions = np.array([model.call(x[np.newaxis, :]).numpy() for x in test_X])
mse = np.mean(np.square(predictions - test_y))
print(f"Test MSE: {mse}")
