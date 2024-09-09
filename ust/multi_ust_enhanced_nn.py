import tensorflow as tf
from adaptive_multi_ust import AdaptiveMultiUST

class USTLayer(tf.keras.layers.Layer):
    def __init__(self, ust_dimensions, **kwargs):
        super(USTLayer, self).__init__(**kwargs)
        self.ust = AdaptiveMultiUST(dimensions=ust_dimensions)
        self.ust_dimensions = ust_dimensions

    def build(self, input_shape):
        for i in range(input_shape[1]):
            self.ust.add_node([i] * self.ust_dimensions, i)

    def call(self, inputs):
        processed_inputs = []
        for i in range(inputs.shape[1]):
            nearest, _ = self.ust.find_nearest([i] * self.ust_dimensions)
            processed_inputs.append(inputs[:, i] * (nearest.data + 1) / inputs.shape[1])
        return tf.stack(processed_inputs, axis=1)

# Example usage
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20,)),
    USTLayer(ust_dimensions=3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Generate some sample data
X = tf.random.normal((1000, 20))
y = tf.reduce_sum(X, axis=1, keepdims=True)

# Train the model
history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model
test_X = tf.random.normal((100, 20))
test_y = tf.reduce_sum(test_X, axis=1, keepdims=True)
test_loss = model.evaluate(test_X, test_y)
print(f"Test loss: {test_loss}")
