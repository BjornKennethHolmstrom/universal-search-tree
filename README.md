# Universal Search Tree (UST)

## Project Overview
The Universal Search Tree (UST) is a flexible and efficient data structure for complex tree-based searches with unlimited connectivity. This project explores various implementations and enhancements of the UST concept, including neural network-guided adaptations and hierarchical models.

## Features
- Dynamic node creation and connection
- Tree cutting and merging
- Bidirectional connections between nodes
- Shortest path finding between nodes
- Visualization of tree structures
- Neural network-guided adaptive UST
- MultiUST-enhanced neural networks
- Hierarchical Neural-UST model

## Project Structure
```
ust
├── docs
│   ├── adaptive-multi-ust-docs.md
│   ├── combined-nn-ust-models-doc.md
│   ├── hierarchical-neural-ust-doc.md
│   ├── images
│   │   ├── hierarchical_neural_ust_visualization.png
│   │   ├── multiust_enhanced_nn_visualization.png
│   │   ├── nn_guided_multiust_visualization_after.png
│   │   ├── nn_guided_multiust_visualization_before.png
│   │   └── subnet_activations_visualization.png
│   ├── multiust-enhanced-nn-doc.md
│   ├── neural-ust-hybrid.md
│   ├── nn-guided-multiust-doc.md
│   └── ust-optimization-strategies.md
├── README.md
├── requirements.txt
├── test_ust.py
└── ust
    ├── adaptive_multi_ust.py
    ├── algorithms.py
    ├── core.py
    ├── hierarchical_neural_ust_model.py
    ├── __init__.py
    ├── multi_ust_enhanced_nn.py
    ├── nn_guided_adaptive_multi_ust.py
    ├── test_utils.py
    ├── utils.py
    ├── visualize_hierarchical_neural_ust_model.py
    ├── visualize_multi_ust_enhanced_nn.py
    └── visualize_nn_guided_multiust.py


```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/universal-search-tree.git
   cd universal-search-tree
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python3 -m venv ust_env
   source ust_env/bin/activate  # On Windows use `ust_env\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
For basic usage of the Universal Search Tree, refer to the example in the original README.

For information on the new models and visualizations, please see the combined documentation in `docs/combined_ust_models_doc.md`.

## Running Tests
To run the test suite:

```
python3 test_ust.py
```

## Visualizations
To generate visualizations for the new models:

```
python ust/visualize_nn_guided_multiust.py
python ust/visualize_multiust_enhanced_nn.py
python ust/visualize_hierarchical_neural_ust.py
```

The resulting PNG files will be saved in the project root directory.

## Documentation
For a comprehensive overview of all UST models and their visualizations, see `docs/combined_ust_models_doc.md`.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
