import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from hierarchical_neural_ust_model import HierarchicalNeuralUST

def visualize_hierarchical_ust(model, title="Hierarchical Neural-UST Model"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize UST nodes (subnets)
    for i, node in enumerate(model.ust.nodes):
        coords = node.coords
        ax.scatter(*coords, c='lightblue', s=100)
        ax.text(*coords, f'Subnet {i}', fontsize=8)
    
    # Visualize connections
    for node in model.ust.nodes:
        for dim in range(model.ust.dimensions):
            for connected_node in node.connections[dim]:
                ax.plot(*zip(node.coords, connected_node.coords), c='gray', alpha=0.2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('hierarchical_neural_ust_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_subnet_activations(model, input_data, title="Subnet Activations"):
    activations = []
    for x in input_data:
        try:
            nearest, _ = model.ust.find_nearest(np.random.rand(model.ust.dimensions))
            activations.append(nearest.coords)
        except Exception as e:
            print(f"Error finding nearest node: {e}")
            continue
    
    if not activations:
        print("No valid activations to visualize.")
        return
    
    activations = np.array(activations)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(activations[:, 0], activations[:, 1], activations[:, 2], c=range(len(activations)), cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(scatter, label='Input Sample Index')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('subnet_activations_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create a sample model
input_dim = 10
output_dim = 1
ust_dimensions = 3
num_subnets = 5
model = HierarchicalNeuralUST(input_dim, output_dim, ust_dimensions, num_subnets)

# Visualize the hierarchical UST structure
visualize_hierarchical_ust(model)

# Generate sample input data
X = np.random.normal(size=(100, input_dim)).astype(np.float32)

# Visualize subnet activations for the sample input data
visualize_subnet_activations(model, X)

print("Visualization complete. Check the generated PNG files.")
