import os
import sys

# Debug prints
print("Current directory:", os.getcwd())
print("Python path:", sys.path)
print("Directory contents:", os.listdir('.'))
print("UST directory contents:", os.listdir('./ust'))

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import unittest
from ust.academic.academic_ust import AcademicUST

class TestAcademicUST(unittest.TestCase):
    def setUp(self):
        self.graph = AcademicUST()
        
        # Add sample data
        self.graph.add_publication(
            title="Understanding Universal Search Trees",
            authors=["Alice Smith", "Bob Jones"],
            year=2023,
            venue="Journal of Advanced Data Structures",
            topics=["Search Trees", "Algorithm Optimization", "Data Structures"],
            abstract="This paper introduces the concept of Universal Search Trees...",
            citations=15
        )

    def test_basic_functionality(self):
        self.assertIn("Understanding Universal Search Trees", self.graph.publications)
        self.assertIn("Alice Smith", self.graph.authors)

if __name__ == '__main__':
    unittest.main()
