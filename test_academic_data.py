import unittest
from ust.academic.academic_ust import AcademicUST
import random
from typing import List, Dict, Set
import matplotlib.pyplot as plt
import networkx as nx
from math import sqrt
from ust.academic.academic_analytics import AcademicAnalytics, print_analytics_report

class AcademicTestData:
    def __init__(self):
        # Define test data parameters
        self.institutions = [
            "Stanford University",
            "MIT",
            "Berkeley",
            "Cambridge University",
            "ETH Zurich"
        ]
        
        self.research_areas = {
            "Machine Learning": [
                "Deep Learning",
                "Reinforcement Learning",
                "Neural Networks",
                "Probabilistic Models"
            ],
            "Mathematics": [
                "Graph Theory",
                "Topology",
                "Number Theory",
                "Abstract Algebra"
            ],
            "Physics": [
                "Quantum Computing",
                "Statistical Mechanics",
                "Particle Physics",
                "Condensed Matter"
            ]
        }
        
        # Generate synthetic authors with research interests
        self.authors = self._generate_authors(30)  # 30 authors
        
        # Generate synthetic publications
        self.publications = self._generate_publications(50)  # 50 papers
    
    def _generate_authors(self, num_authors: int) -> List[Dict]:
        authors = []
        for i in range(num_authors):
            # Assign primary research area
            primary_area = random.choice(list(self.research_areas.keys()))
            
            # Assign some topics from primary area and maybe other areas
            topics = set()
            # Add 2-3 topics from primary area
            topics.update(random.sample(self.research_areas[primary_area], 
                                     random.randint(2, 3)))
            # Maybe add 1-2 topics from other areas
            if random.random() < 0.3:  # 30% chance
                other_area = random.choice([a for a in self.research_areas.keys() 
                                         if a != primary_area])
                topics.update(random.sample(self.research_areas[other_area], 
                                         random.randint(1, 2)))
            
            authors.append({
                "name": f"Author_{i+1}",
                "institution": random.choice(self.institutions),
                "h_index": random.randint(1, 30),
                "research_areas": list(topics),
                "primary_area": primary_area
            })
        return authors
    
    def _generate_publications(self, num_papers: int) -> List[Dict]:
        publications = []
        # Track citations to ensure realistic citation patterns
        years = list(range(2020, 2025))
        earlier_papers = []
        
        for i in range(num_papers):
            # Select authors (1-4 authors per paper)
            num_authors = random.choices([1, 2, 3, 4], weights=[0.1, 0.4, 0.3, 0.2])[0]
            
            # Try to select authors from similar research areas
            primary_author = random.choice(self.authors)
            authors = [primary_author]
            
            # Add co-authors with similar interests
            potential_coauthors = [
                a for a in self.authors 
                if a != primary_author and 
                (a["primary_area"] == primary_author["primary_area"] or 
                 any(topic in a["research_areas"] 
                     for topic in primary_author["research_areas"]))
            ]
            
            if potential_coauthors and num_authors > 1:
                coauthors = random.sample(potential_coauthors, 
                                        min(num_authors - 1, len(potential_coauthors)))
                authors.extend(coauthors)
            
            # Select topics based on authors' interests
            all_topics = set()
            for author in authors:
                all_topics.update(author["research_areas"])
            paper_topics = random.sample(list(all_topics), 
                                       min(random.randint(2, 4), len(all_topics)))
            
            # Generate citations
            year = random.choice(years)
            citations = 0
            if earlier_papers and year > min(years):
                # More recent papers have fewer citations
                max_citations = (year - min(years)) * 20
                citations = random.randint(0, max_citations)
            
            paper = {
                "title": f"Paper_{i+1}_on_{'_'.join(paper_topics[:2])}",
                "authors": [a["name"] for a in authors],
                "year": year,
                "venue": f"Journal of {primary_author['primary_area']}",
                "topics": paper_topics,
                "abstract": f"This paper investigates {' and '.join(paper_topics)}...",
                "citations": citations
            }
            
            publications.append(paper)
            if year < max(years):
                earlier_papers.append(paper)
        
        return publications
    
    def populate_ust(self, ust: AcademicUST):
        """Populate the Academic UST with synthetic test data."""
        # First add all authors with their details
        for author in self.authors:
            ust.add_author_details(
                name=author["name"],
                institution=author["institution"],
                h_index=author["h_index"]
            )
        
        # Then add all publications with their relationships
        for pub in self.publications:
            ust.add_publication(
                title=pub["title"],
                authors=pub["authors"],
                year=pub["year"],
                venue=pub["venue"],
                topics=pub["topics"],
                abstract=pub["abstract"],
                citations=pub["citations"]
            )
        
        return ust

    def visualize_collaboration_network(self, ust: AcademicUST, filename: str = "collaboration_network.png"):
        """Create and save a visualization of the author collaboration network."""
        G = nx.Graph()
        
        # Add nodes colored by primary research area
        color_map = {
            "Machine Learning": "lightblue",
            "Mathematics": "lightgreen",
            "Physics": "lightcoral"
        }
        
        node_colors = []
        for author in self.authors:
            G.add_node(author["name"])
            node_colors.append(color_map.get(author["primary_area"], "gray"))
        
        # Add edges based on collaborations
        for pub in self.publications:
            authors = pub["authors"]
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]["weight"] += 1
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1/sqrt(G.number_of_nodes()))
        
        # Draw the network
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=1000,
                font_size=8,
                with_labels=True,
                edge_color='gray',
                alpha=0.7)
        
        plt.title("Author Collaboration Network")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return G

class TestAcademicUST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods."""
        cls.test_data = AcademicTestData()
        cls.ust = AcademicUST()
        cls.test_data.populate_ust(cls.ust)
        
        # Create analytics
        cls.analytics = AcademicAnalytics(cls.test_data)
        
        # Generate visualizations
        cls.test_data.visualize_collaboration_network(cls.ust, "collaboration_network.png")
        cls.ust.visualize_graph("academic_test_graph.png")
        cls.analytics.visualize_communities("research_communities.png")
        cls.analytics.visualize_research_trends("research_trends.png")
    
    def test_collaboration_network(self):
        """Test if collaboration network is properly constructed."""
        random_author = random.choice(list(self.ust.authors.keys()))
        collaborators = self.ust.find_collaborators(random_author)
        self.assertIsInstance(collaborators, dict)
        print(f"\nTested collaborations for {random_author}:")
        print(f"Found {len(collaborators)} collaborators")
    
    def test_similar_papers(self):
        """Test paper similarity functionality."""
        random_paper = random.choice(list(self.ust.publications.keys()))
        similar_papers = self.ust.find_similar_papers(random_paper)
        self.assertIsInstance(similar_papers, list)
        print(f"\nTested paper similarity for {random_paper}:")
        print(f"Found {len(similar_papers)} similar papers")
    
    def test_topic_experts(self):
        """Test expert identification functionality."""
        test_topics = list(self.ust.topics)
        if test_topics:
            random_topic = random.choice(test_topics)
            experts = self.ust.get_topic_experts(random_topic)
            self.assertIsInstance(experts, list)
            print(f"\nTested expert identification for topic '{random_topic}':")
            print(f"Found {len(experts)} experts")
    
    def test_basic_statistics(self):
        """Test basic statistics of the generated network."""
        print("\nNetwork Statistics:")
        print(f"Number of authors: {len(self.ust.authors)}")
        print(f"Number of papers: {len(self.ust.publications)}")
        print(f"Number of topics: {len(self.ust.topics)}")
        print(f"Number of institutions: {len(self.ust.institutions)}")
        
        self.assertGreater(len(self.ust.authors), 0)
        self.assertGreater(len(self.ust.publications), 0)
        self.assertGreater(len(self.ust.topics), 0)
        self.assertGreater(len(self.ust.institutions), 0)

    def test_community_detection(self):
        """Test community detection and analysis."""
        communities = self.analytics.detect_communities()
        community_analysis = self.analytics.analyze_community_characteristics()
        
        self.assertGreater(len(communities), 0)
        self.assertEqual(len(communities), len(self.test_data.authors))
        print("\n=== Community Detection Results ===")
        print(f"Number of communities detected: {len(set(communities.values()))}")
    
    def test_research_trends(self):
        """Test research trend analysis."""
        trends = self.analytics.analyze_research_trends()
        
        self.assertGreater(len(trends), 0)
        print("\n=== Research Trends ===")
        print_analytics_report(self.analytics)

def run_test_queries(ust: AcademicUST):
    """Run various test queries on the populated UST."""
    print("\n=== Testing Academic UST Queries ===")
    
    # Test author collaboration network
    print("\n1. Testing author collaborations:")
    test_author = random.choice(list(ust.authors.keys()))
    collaborators = ust.find_collaborators(test_author)
    print(f"Collaborators for {test_author}:")
    for collaborator, distance in collaborators.items():
        print(f"  - {collaborator} (distance: {distance})")
    
    # Test similar papers
    print("\n2. Testing paper similarity:")
    test_paper = random.choice(list(ust.publications.keys()))
    similar_papers = ust.find_similar_papers(test_paper)
    print(f"Papers similar to '{test_paper}':")
    for paper, similarity in similar_papers[:5]:  # Show top 5
        print(f"  - {paper} (similarity: {similarity:.2f})")
    
    # Test topic experts
    print("\n3. Testing topic expert identification:")
    test_topic = random.choice(list(ust.topics))
    experts = ust.get_topic_experts(test_topic)
    print(f"Experts in {test_topic}:")
    for expert, num_papers, total_citations in experts[:5]:  # Show top 5
        print(f"  - {expert} ({num_papers} papers, {total_citations} citations)")
    
    return {
        "test_author": test_author,
        "test_paper": test_paper,
        "test_topic": test_topic
    }

if __name__ == '__main__':
    # If run directly, create visualizations and run tests
    unittest.main(verbosity=2)

