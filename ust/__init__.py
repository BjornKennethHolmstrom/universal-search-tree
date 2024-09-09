# ust/__init__.py

from .core import UniversalSearchTree, USTNode
from .algorithms import find_shortest_path
from .utils import visualize_tree
from .adaptive_multi_ust import AdaptiveMultiUST, MultiDimNode

__all__ = ['UniversalSearchTree', 'USTNode', 'find_shortest_path', 'visualize_tree', 'AdaptiveMultiUST', 'MultiDimNode']

__version__ = '0.1.0'

# Optional: add a brief description of the package
__doc__ = """
Universal Search Tree (UST) implementation.

This package provides a flexible and efficient data structure for 
complex tree-based searches with unlimited connectivity.
"""
