"""
Automatic knowledge graph construction from unstructured text.
Handles graph database operations and entity-relationship modeling.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

from .ontology_generator import Entity, Relationship, OntologySchema

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any]
    embedding_id: Optional[str] = None


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relationship_type: str
    properties: Dict[str, Any]
    weight: float = 1.0


class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs from entity data."""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", qdrant_api_key: str = None):
        self.graph = nx.MultiDiGraph()
        
        # Initialize Qdrant client with optional API key for cloud instances
        if qdrant_api_key:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.qdrant_client = QdrantClient(url=qdrant_url)
            
        self.collection_name = "entity_embeddings"
        self._setup_qdrant_collection()
        
    def _setup_qdrant_collection(self):
        """Setup Qdrant collection for entity embeddings."""
        collections = self.qdrant_client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def add_entity(self, entity: Entity) -> GraphNode:
        """Add an entity to the knowledge graph."""
        node_id = f"entity_{entity.id}"
        
        # Create graph node
        node = GraphNode(
            id=node_id,
            label=entity.type,
            properties={
                'name': entity.name,
                'description': entity.description,
                'original_id': entity.id,
                'entity_type': entity.type,
                **entity.properties
            }
        )
        
        # Add to NetworkX graph
        self.graph.add_node(
            node_id,
            label=entity.type,
            name=entity.name,
            description=entity.description,
            **entity.properties
        )
        
        # Store embedding in Qdrant
        if entity.embeddings:
            embedding_id = str(uuid.uuid4())
            node.embedding_id = embedding_id
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=embedding_id,
                        vector=entity.embeddings,
                        payload={
                            'node_id': node_id,
                            'entity_id': entity.id,
                            'name': entity.name,
                            'type': entity.type,
                            'description': entity.description
                        }
                    )
                ]
            )
        
        logger.info(f"Added entity: {entity.name} ({entity.type})")
        return node
    
    def add_relationship(self, relationship: Relationship) -> GraphEdge:
        """Add a relationship to the knowledge graph."""
        source_id = f"entity_{relationship.source_id}"
        target_id = f"entity_{relationship.target_id}"
        
        # Create graph edge
        edge = GraphEdge(
            source=source_id,
            target=target_id,
            relationship_type=relationship.relationship_type,
            properties=relationship.properties,
            weight=relationship.confidence
        )
        
        # Add to NetworkX graph
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship.relationship_type,
            weight=relationship.confidence,
            **relationship.properties
        )
        
        logger.info(f"Added relationship: {source_id} --{relationship.relationship_type}--> {target_id}")
        return edge
    
    def build_from_entities(self, entities: List[Entity], relationships: List[Relationship]) -> None:
        """Build knowledge graph from entities and relationships."""
        logger.info(f"Building knowledge graph from {len(entities)} entities and {len(relationships)} relationships")
        
        # Add all entities
        for entity in entities:
            self.add_entity(entity)
        
        # Add all relationships
        for relationship in relationships:
            # Only add if both entities exist
            source_exists = f"entity_{relationship.source_id}" in self.graph.nodes
            target_exists = f"entity_{relationship.target_id}" in self.graph.nodes
            
            if source_exists and target_exists:
                self.add_relationship(relationship)
            else:
                logger.warning(f"Skipping relationship {relationship.source_id} -> {relationship.target_id} (missing entities)")
    
    def get_entity_neighbors(self, entity_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get neighbors of an entity up to specified depth."""
        node_id = f"entity_{entity_id}"
        
        if node_id not in self.graph.nodes:
            return []
        
        neighbors = []
        for depth in range(1, max_depth + 1):
            try:
                # Get nodes at specific depth
                nodes_at_depth = list(nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth))
                nodes_at_depth = [n for n in nodes_at_depth if n != node_id]
                
                for neighbor_id in nodes_at_depth:
                    if neighbor_id.startswith('entity_'):
                        neighbor_data = self.graph.nodes[neighbor_id]
                        neighbors.append({
                            'id': neighbor_id,
                            'name': neighbor_data.get('name', ''),
                            'type': neighbor_data.get('label', ''),
                            'depth': depth,
                            'description': neighbor_data.get('description', '')[:200]
                        })
            except nx.NetworkXError:
                break
        
        return neighbors
    
    def find_path(self, source_id: str, target_id: str, max_length: int = 5) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        source_node = f"entity_{source_id}"
        target_node = f"entity_{target_id}"
        
        try:
            path = nx.shortest_path(self.graph, source_node, target_node)
            return path if len(path) <= max_length + 1 else None
        except nx.NetworkXNoPath:
            return None
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_types': list(set(data.get('label', 'Unknown') for _, data in self.graph.nodes(data=True))),
            'relationship_types': list(set(data.get('relationship_type', 'Unknown') for _, _, data in self.graph.edges(data=True))),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
    
    def export_to_cypher(self, limit: Optional[int] = None) -> List[str]:
        """Export graph to Cypher queries for Neo4j."""
        cypher_queries = []
        
        # Create nodes
        node_count = 0
        for node_id, data in self.graph.nodes(data=True):
            if limit and node_count >= limit:
                break
                
            cypher = f"CREATE (n:{data.get('label', 'Entity')} {{"
            properties = []
            for key, value in data.items():
                if key != 'label':
                    if isinstance(value, str):
                        escaped_value = value.replace("'", "\\'")
                        properties.append(f"{key}: '{escaped_value}'")
                    else:
                        properties.append(f"{key}: {json.dumps(value)}")
            cypher += ", ".join(properties) + "})"
            cypher_queries.append(cypher)
            node_count += 1
        
        # Create relationships
        edge_count = 0
        for source, target, data in self.graph.edges(data=True):
            if limit and edge_count >= limit:
                break
                
            rel_type = data.get('relationship_type', 'RELATED_TO')
            cypher = f"MATCH (a), (b) WHERE a.id = '{source}' AND b.id = '{target}' CREATE (a)-[r:{rel_type}]->(b)"
            cypher_queries.append(cypher)
            edge_count += 1
        
        return cypher_queries
    
    def search_similar_entities(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar entities using vector similarity."""
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for result in search_results:
            results.append({
                'id': result.payload['entity_id'],
                'name': result.payload['name'],
                'type': result.payload['type'],
                'description': result.payload['description'],
                'score': result.score
            })
        
        return results


if __name__ == "__main__":
    # Test graph builder
    from .ontology_generator import Entity
    
    builder = KnowledgeGraphBuilder()
    
    # Create test entities
    entity1 = Entity(
        id="test_1",
        name="Albert Einstein",
        type="Person",
        description="Theoretical physicist",
        properties={},
        embeddings=[0.1] * 1024
    )
    
    entity2 = Entity(
        id="test_2", 
        name="Princeton University",
        type="Organization",
        description="University where Einstein worked",
        properties={},
        embeddings=[0.2] * 1024
    )
    
    # Add entities to graph
    builder.add_entity(entity1)
    builder.add_entity(entity2)
    
    # Add relationship
    from .ontology_generator import Relationship
    relationship = Relationship(
        source_id="test_1",
        target_id="test_2", 
        relationship_type="worked_at",
        confidence=0.9,
        properties={}
    )
    
    builder.add_relationship(relationship)
    
    # Get statistics
    stats = builder.get_graph_statistics()
    print(f"Graph statistics: {stats}")
