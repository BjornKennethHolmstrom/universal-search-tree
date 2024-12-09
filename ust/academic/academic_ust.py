# ust/academic/academic_ust.py

from typing import List, Set, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from .semantic_ust import SemanticUST, SemanticNode

@dataclass
class Publication:
    title: str
    year: int
    venue: str
    abstract: str
    citations: int = 0

@dataclass
class Author:
    name: str
    institution: str
    h_index: Optional[int] = None

class AcademicUST(SemanticUST):
    """Specialized Semantic UST for academic knowledge graphs."""
    
    def __init__(self):
        super().__init__()
        self.publications: Dict[str, Publication] = {}
        self.authors: Dict[str, Author] = {}
        self.topics: Set[str] = set()
        self.institutions: Set[str] = set()
    
    def add_publication(self, 
                       title: str, 
                       authors: List[str], 
                       year: int,
                       venue: str,
                       topics: List[str],
                       abstract: str,
                       citations: int = 0):
        """Add a publication with its relationships to authors and topics."""
        pub = Publication(title, year, venue, abstract, citations)
        self.publications[title] = pub
        self.add_node(title, {
            "type": "publication",
            "year": str(year),
            "venue": venue,
            "citations": str(citations)
        })
        
        for author_name in authors:
            if author_name not in self.authors:
                self.authors[author_name] = Author(author_name, "")
                self.add_node(author_name, {"type": "author"})
            
            self.add_relationship(title, author_name, "authored_by")
        
        for topic in topics:
            self.topics.add(topic)
            self.add_node(topic, {"type": "topic"})
            self.add_relationship(title, topic, "belongs_to_topic")
    
    def add_author_details(self, name: str, institution: str, h_index: Optional[int] = None):
        if name not in self.authors:
            self.authors[name] = Author(name, institution, h_index)
            self.add_node(name, {"type": "author"})
        else:
            self.authors[name].institution = institution
            self.authors[name].h_index = h_index
        
        self.institutions.add(institution)
        self.add_node(institution, {"type": "institution"})
        self.add_relationship(name, institution, "affiliated_with")
    
    def find_collaborators(self, author_name: str, max_distance: int = 2) -> Dict[str, int]:
        collaborators = {}
        visited = {author_name}
        current_level = {author_name}
        distance = 0
        
        while distance < max_distance and current_level:
            next_level = set()
            for author in current_level:
                papers = self.get_related_entities(author, "inverse_authored_by")
                
                for paper in papers:
                    coauthors = self.get_related_entities(paper, "authored_by")
                    for coauthor in coauthors:
                        if coauthor not in visited:
                            collaborators[coauthor] = distance + 1
                            next_level.add(coauthor)
                            visited.add(coauthor)
            
            current_level = next_level
            distance += 1
        
        return collaborators
    
    def find_similar_papers(self, paper_title: str) -> List[tuple]:
        if paper_title not in self.publications:
            return []
        
        topics = self.get_related_entities(paper_title, "belongs_to_topic")
        authors = self.get_related_entities(paper_title, "authored_by")
        
        similar_papers = []
        for other_title, pub in self.publications.items():
            if other_title != paper_title:
                other_topics = self.get_related_entities(other_title, "belongs_to_topic")
                other_authors = self.get_related_entities(other_title, "authored_by")
                
                topic_overlap = len(topics & other_topics)
                author_overlap = len(authors & other_authors)
                
                if topic_overlap > 0 or author_overlap > 0:
                    score = (topic_overlap * 2 + author_overlap * 3) / (len(topics) + len(authors))
                    similar_papers.append((other_title, score))
        
        return sorted(similar_papers, key=lambda x: x[1], reverse=True)
    
    def get_topic_experts(self, topic: str, min_papers: int = 2) -> List[tuple]:
        if topic not in self.topics:
            return []
        
        topic_papers = self.get_related_entities(topic, "inverse_belongs_to_topic")
        author_metrics = defaultdict(lambda: {"papers": 0, "citations": 0})
        
        for paper in topic_papers:
            authors = self.get_related_entities(paper, "authored_by")
            citations = int(self.nodes[paper].properties.get("citations", 0))
            
            for author in authors:
                author_metrics[author]["papers"] += 1
                author_metrics[author]["citations"] += citations
        
        experts = [
            (author, metrics["papers"], metrics["citations"])
            for author, metrics in author_metrics.items()
            if metrics["papers"] >= min_papers
        ]
        
        return sorted(experts, key=lambda x: (x[1], x[2]), reverse=True)

    def visualize_graph(self, filename: str = "academic_graph.png"):
        """Visualize the academic knowledge graph."""
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.Graph()
        colors = []
        
        color_map = {
            "publication": "lightblue",
            "author": "lightgreen",
            "topic": "lightcoral",
            "institution": "lightyellow"
        }
        
        for node_id, node in self.nodes.items():
            G.add_node(node_id)
            colors.append(color_map.get(node.properties.get("type", ""), "gray"))
        
        for node_id, node in self.nodes.items():
            for rel_type, connected_nodes in node.relationships.items():
                for connected_node in connected_nodes:
                    G.add_edge(node_id, connected_node.entity, relationship=rel_type)
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=colors, with_labels=True, node_size=500, font_size=8)
        plt.title("Academic Knowledge Graph")
        plt.savefig(filename)
        plt.close()
