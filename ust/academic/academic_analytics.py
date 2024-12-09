import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
from community import community_louvain  # python-louvain package
import seaborn as sns
from datetime import datetime

class AcademicAnalytics:
    def __init__(self, ust_data):
        self.test_data = ust_data
        self.collaboration_graph = None
        self.communities = None
        
    def build_collaboration_graph(self) -> nx.Graph:
        """Build and analyze the collaboration network."""
        G = nx.Graph()
        
        # Add nodes with attributes
        for author in self.test_data.authors:
            G.add_node(author["name"], 
                      primary_area=author["primary_area"],
                      research_areas=author["research_areas"],
                      h_index=author["h_index"],
                      institution=author["institution"])
        
        # Add edges with weights based on collaboration frequency
        edge_weights = defaultdict(int)
        for pub in self.test_data.publications:
            authors = pub["authors"]
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    edge_weights[(authors[i], authors[j])] += 1
        
        # Add weighted edges to graph
        for (author1, author2), weight in edge_weights.items():
            G.add_edge(author1, author2, weight=weight)
        
        self.collaboration_graph = G
        return G
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect research communities using the Louvain method."""
        if not self.collaboration_graph:
            self.build_collaboration_graph()
        
        self.communities = community_louvain.best_partition(self.collaboration_graph)
        return self.communities
    
    def analyze_community_characteristics(self) -> List[Dict]:
        """Analyze the characteristics of each detected community."""
        if not self.communities:
            self.detect_communities()
        
        community_data = defaultdict(lambda: {
            "members": [],
            "primary_areas": Counter(),
            "research_topics": Counter(),
            "avg_h_index": 0,
            "institutions": Counter(),
            "papers": []
        })
        
        # Gather community member data
        for author, community_id in self.communities.items():
            author_data = next(a for a in self.test_data.authors if a["name"] == author)
            community_data[community_id]["members"].append(author)
            community_data[community_id]["primary_areas"][author_data["primary_area"]] += 1
            community_data[community_id]["institutions"][author_data["institution"]] += 1
            for topic in author_data["research_areas"]:
                community_data[community_id]["research_topics"][topic] += 1
            
        # Calculate average h-index and find papers
        for comm_id, data in community_data.items():
            h_indices = [a["h_index"] for a in self.test_data.authors 
                        if a["name"] in data["members"]]
            data["avg_h_index"] = np.mean(h_indices)
            
            data["papers"] = [p for p in self.test_data.publications 
                            if any(author in data["members"] for author in p["authors"])]
        
        return dict(community_data)
    
    def analyze_research_trends(self) -> Dict:
        """Analyze research trends over time."""
        trends = defaultdict(lambda: defaultdict(int))
        yearly_citations = defaultdict(lambda: defaultdict(int))
        
        # Track topic popularity and impact over time
        for pub in self.test_data.publications:
            year = pub["year"]
            for topic in pub["topics"]:
                trends[topic][year] += 1
                yearly_citations[topic][year] += pub["citations"]
        
        # Calculate trend metrics
        trend_metrics = {}
        for topic in trends.keys():
            years = sorted(trends[topic].keys())
            if len(years) > 1:
                # Calculate growth rate
                initial_papers = trends[topic][years[0]]
                final_papers = trends[topic][years[-1]]
                growth_rate = (final_papers - initial_papers) / initial_papers if initial_papers > 0 else 0
                
                # Calculate impact growth
                initial_impact = yearly_citations[topic][years[0]]
                final_impact = yearly_citations[topic][years[-1]]
                impact_growth = (final_impact - initial_impact) / initial_impact if initial_impact > 0 else 0
                
                trend_metrics[topic] = {
                    "total_papers": sum(trends[topic].values()),
                    "total_citations": sum(yearly_citations[topic].values()),
                    "growth_rate": growth_rate,
                    "impact_growth": impact_growth,
                    "yearly_papers": dict(trends[topic]),
                    "yearly_citations": dict(yearly_citations[topic])
                }
        
        return trend_metrics
    
    def visualize_communities(self, filename: str = "community_visualization.png"):
        """Create a visualization of research communities."""
        if not self.collaboration_graph or not self.communities:
            self.detect_communities()
        
        plt.figure(figsize=(15, 10))
        
        # Position nodes using force-directed layout
        pos = nx.spring_layout(self.collaboration_graph)
        
        # Draw nodes colored by community
        num_communities = len(set(self.communities.values()))
        colors = plt.cm.rainbow(np.linspace(0, 1, num_communities))
        
        for comm_id in range(num_communities):
            node_list = [node for node, community in self.communities.items() 
                        if community == comm_id]
            nx.draw_networkx_nodes(self.collaboration_graph, pos,
                                 nodelist=node_list,
                                 node_color=[colors[comm_id]],
                                 node_size=500,
                                 alpha=0.6,
                                 label=f"Community {comm_id}")
        
        # Draw edges with varying thickness based on weight
        edges = self.collaboration_graph.edges(data=True)
        weights = [data["weight"] for _, _, data in edges]
        nx.draw_networkx_edges(self.collaboration_graph, pos,
                             width=[w/max(weights) * 2 for w in weights],
                             alpha=0.3)
        
        # Add labels
        nx.draw_networkx_labels(self.collaboration_graph, pos, font_size=8)
        
        plt.title("Research Communities and Collaboration Network")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    def visualize_research_trends(self, filename: str = "research_trends.png"):
        """Visualize research trends over time."""
        trends = self.analyze_research_trends()
        
        # Create a multi-panel visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Topic popularity over time
        for topic, metrics in trends.items():
            years = sorted(metrics["yearly_papers"].keys())
            papers = [metrics["yearly_papers"][year] for year in years]
            ax1.plot(years, papers, marker='o', label=topic)
        
        ax1.set_title("Research Topic Popularity Over Time")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Number of Papers")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Topic impact (citations) over time
        for topic, metrics in trends.items():
            years = sorted(metrics["yearly_citations"].keys())
            citations = [metrics["yearly_citations"][year] for year in years]
            ax2.plot(years, citations, marker='s', label=topic)
        
        ax2.set_title("Research Topic Impact Over Time")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Number of Citations")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

def print_analytics_report(analytics: AcademicAnalytics):
    """Print a comprehensive analytics report."""
    print("\n=== Research Community Analysis Report ===\n")
    
    # Community Analysis
    communities = analytics.analyze_community_characteristics()
    print(f"Detected {len(communities)} distinct research communities:\n")
    
    for comm_id, data in communities.items():
        print(f"Community {comm_id}:")
        print(f"  Members: {len(data['members'])} researchers")
        print("  Primary Research Areas:", 
              dict(data['primary_areas'].most_common(3)))
        print("  Top Research Topics:", 
              dict(data['research_topics'].most_common(3)))
        print(f"  Average h-index: {data['avg_h_index']:.2f}")
        print(f"  Publications: {len(data['papers'])}")
        print("  Institutions:", dict(data['institutions'].most_common(3)))
        print()
    
    print("\n=== Research Trends Analysis ===\n")
    
    trends = analytics.analyze_research_trends()
    print("Top Growing Research Topics:")
    sorted_topics = sorted(trends.items(), 
                         key=lambda x: x[1]['growth_rate'], 
                         reverse=True)
    
    for topic, metrics in sorted_topics[:5]:
        print(f"\n{topic}:")
        print(f"  Total Papers: {metrics['total_papers']}")
        print(f"  Total Citations: {metrics['total_citations']}")
        print(f"  Growth Rate: {metrics['growth_rate']:.2%}")
        print(f"  Impact Growth: {metrics['impact_growth']:.2%}")
