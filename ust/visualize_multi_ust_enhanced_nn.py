import matplotlib.pyplot as plt
import numpy as np
from multi_ust_enhanced_nn import USTLayer, model
import tensorflow as tf

def visualize_ust_effect(model, input_data, title="MultiUST-Enhanced Neural Network"):
    # Get the UST layer
    ust_layer = model.layers[0]
    
    # Get the output of the UST layer
    ust_output = ust_layer(input_data).numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Visualize input data
    ax1.scatter(input_data[:, 0], input_data[:, 1], c=input_data[:, 2], cmap='viridis')
    ax1.set_title("Input Data")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    
    # Visualize UST layer output
    scatter = ax2.scatter(ust_output[:, 0], ust_output[:, 1], c=ust_output[:, 2], cmap='viridis')
    ax2.set_title("UST Layer Output")
    ax2.set_xlabel("Transformed Feature 1")
    ax2.set_ylabel("Transformed Feature 2")
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax2, label="Feature 3")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('multiust_enhanced_nn_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate sample data
np.random.seed(42)
X = np.random.normal(size=(1000, 20)).astype(np.float32)

# Visualize the effect of the UST layer
visualize_ust_effect(model, X, "MultiUST-Enhanced Neural Network - UST Layer Effect")

# Train the model (assuming you have target data 'y')
# y = np.sum(X, axis=1, keepdims=True).astype(np.float32)
# history = model.fit(X, y, epochs=10, validation_split=0.2, verbose=0)

# Visualize the effect of the UST layer after training
# visualize_ust_effect(model, X, "MultiUST-Enhanced Neural Network - UST Layer Effect After Training")
