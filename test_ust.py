import unittest
from ust.core import UniversalSearchTree
from ust.test_utils import generate_random_ust, verify_ust_properties, compare_paths

class TestUniversalSearchTree(unittest.TestCase):

    def setUp(self):
        self.ust = UniversalSearchTree()

    def test_add_node(self):
        self.ust.add_node('A')
        self.assertIn('A', self.ust.nodes)
        self.assertEqual(len(self.ust.nodes), 1)

    def test_connect_nodes(self):
        self.ust.connect_nodes('A', 'B')
        self.assertIn('A', self.ust.nodes)
        self.assertIn('B', self.ust.nodes)
        self.assertIn(self.ust.nodes['B'], self.ust.nodes['A'].children)
        self.assertIn(self.ust.nodes['A'], self.ust.nodes['B'].parents)

    def test_cut_tree(self):
        self.ust.connect_nodes('A', 'B')
        self.ust.connect_nodes('A', 'C')
        self.ust.cut_tree('B')
        self.assertNotIn(self.ust.nodes['B'], self.ust.nodes['A'].children)
        self.assertEqual(len(self.ust.nodes['B'].parents), 0)

    def test_merge_trees(self):
        self.ust.connect_nodes('A', 'B')
        self.ust.connect_nodes('C', 'D')
        self.ust.merge_trees('A', 'C')
        self.assertIn(self.ust.nodes['C'], self.ust.nodes['A'].children)
        self.assertIn(self.ust.nodes['A'], self.ust.nodes['C'].parents)

    def test_find_shortest_path(self):
        self.ust.connect_nodes('A', 'B')
        self.ust.connect_nodes('B', 'C')
        self.ust.connect_nodes('A', 'D')
        self.ust.connect_nodes('D', 'C')
        path = self.ust.find_shortest_path('A', 'C')
        self.assertTrue(compare_paths(path, ['A', 'B', 'C']) or compare_paths(path, ['A', 'D', 'C']))
        self.assertEqual(len(path), 3)  # Ensure the path length is correct

    def test_find_shortest_path_multiple_options(self):
        self.ust.connect_nodes('A', 'B')
        self.ust.connect_nodes('A', 'C')
        self.ust.connect_nodes('B', 'D')
        self.ust.connect_nodes('C', 'D')
        path = self.ust.find_shortest_path('A', 'D')
        self.assertTrue(compare_paths(path, ['A', 'B', 'D']) or compare_paths(path, ['A', 'C', 'D']))
        self.assertEqual(len(path), 3)

    def test_random_ust_properties(self):
        random_ust = generate_random_ust(num_nodes=20, max_children=5)
        self.assertTrue(verify_ust_properties(random_ust))

    def test_leaf_node_property(self):
        self.ust.connect_nodes('A', 'B')
        self.ust.connect_nodes('A', 'C')
        self.assertFalse(self.ust.nodes['A'].is_leaf)
        self.assertTrue(self.ust.nodes['B'].is_leaf)
        self.assertTrue(self.ust.nodes['C'].is_leaf)

    def test_multiple_parents(self):
        self.ust.connect_nodes('A', 'C')
        self.ust.connect_nodes('B', 'C')
        self.assertEqual(len(self.ust.nodes['C'].parents), 2)

if __name__ == '__main__':
    unittest.main()
