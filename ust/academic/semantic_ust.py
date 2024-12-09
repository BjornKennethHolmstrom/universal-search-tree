# ust/academic/semantic_ust.py

from typing import Dict, Set, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass(frozen=True)
class SemanticNodeRef:
    """A hashable reference to a SemanticNode."""
    entity: str

    def __hash__(self):
        return hash(self.entity)

@dataclass
class SemanticNode:
    """A node in the Semantic Universal Search Tree."""
    entity: str
    properties: Dict[str, str] = field(default_factory=dict)
    relationships: Dict[str, Set['SemanticNodeRef']] = field(default_factory=lambda: defaultdict(set))
    access_count: int = 0
    common_queries: Dict[str, int] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.entity)

    def __eq__(self, other):
        if not isinstance(other, SemanticNode):
            return False
        return self.entity == other.entity

class SemanticUST:
    """Universal Search Tree adapted for knowledge graph navigation."""
    
    def __init__(self):
        self.nodes: Dict[str, SemanticNode] = {}
        self.relationship_types: Set[str] = set()
    
    def add_node(self, entity: str, properties: Optional[Dict[str, str]] = None) -> SemanticNode:
        if entity not in self.nodes:
            self.nodes[entity] = SemanticNode(entity, properties or {})
        return self.nodes[entity]
    
    def add_relationship(self, entity1: str, entity2: str, relationship_type: str):
        node1 = self.add_node(entity1)
        node2 = self.add_node(entity2)
        
        # Store references instead of actual nodes
        node1.relationships[relationship_type].add(SemanticNodeRef(node2.entity))
        node2.relationships[f"inverse_{relationship_type}"].add(SemanticNodeRef(node1.entity))
        
        self.relationship_types.add(relationship_type)
        self.relationship_types.add(f"inverse_{relationship_type}")
    
    def find_path(self, start_entity: str, end_entity: str, relationship_types: Optional[Set[str]] = None) -> Optional[List[tuple]]:
        if start_entity not in self.nodes or end_entity not in self.nodes:
            return None
            
        start = self.nodes[start_entity]
        end = self.nodes[end_entity]
        
        query_key = f"path:{start_entity}->{end_entity}"
        start.common_queries[query_key] = start.common_queries.get(query_key, 0) + 1
        
        visited = set()
        queue = [(start, [(start, None, None)])]
        
        while queue:
            current, path = queue.pop(0)
            if current not in visited:
                visited.add(current)
                current.access_count += 1
                
                if current == end:
                    return [(node.entity, rel_type) for node, rel_type, _ in path[1:]]
                
                for rel_type, connected_refs in current.relationships.items():
                    if relationship_types is None or rel_type in relationship_types:
                        for node_ref in connected_refs:
                            node = self.nodes[node_ref.entity]
                            if node not in visited:
                                new_path = path + [(node, rel_type, current)]
                                queue.append((node, new_path))
        
        return None
    
    def get_related_entities(self, entity: str, relationship_type: Optional[str] = None) -> Set[str]:
        if entity not in self.nodes:
            return set()
            
        node = self.nodes[entity]
        related = set()
        
        if relationship_type:
            related.update(ref.entity for ref in node.relationships.get(relationship_type, set()))
        else:
            for connected_refs in node.relationships.values():
                related.update(ref.entity for ref in connected_refs)
        
        return related
    
    def optimize_structure(self, min_access_count: int = 10):
        for node in self.nodes.values():
            if node.access_count >= min_access_count:
                common_queries = sorted(
                    node.common_queries.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for query, count in common_queries:
                    if count >= min_access_count:
                        if query.startswith("path:"):
                            _, entities = query.split(":", 1)
                            start, end = entities.split("->")
                            if start in self.nodes and end in self.nodes:
                                self.add_relationship(
                                    start, 
                                    end, 
                                    "optimized_path"
                                )
        
        for node in self.nodes.values():
            node.access_count = 0
            node.common_queries.clear()
