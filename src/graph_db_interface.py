"""
Extensible interface compatible with Neo4j and AWS Neptune.
Provides unified API for different graph database backends.
"""

from typing import Dict, List, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

try:
    from py2neo import Graph as Neo4jGraph
except ImportError:
    Neo4jGraph = None

try:
    from gremlin_python.driver import client, protocol, serializer
except ImportError:
    client = None
    protocol = None
    serializer = None

logger = logging.getLogger(__name__)


@dataclass
class NodeData:
    """Unified node data structure."""
    id: str
    label: str
    properties: Dict[str, Any]


@dataclass
class EdgeData:
    """Unified edge data structure."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]


@dataclass
class QueryResult:
    """Unified query result structure."""
    nodes: List[NodeData]
    edges: List[EdgeData]
    execution_time: float
    metadata: Dict[str, Any]


class GraphDatabaseInterface(ABC):
    """Abstract base class for graph database interfaces."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the graph database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the graph database."""
        pass
    
    @abstractmethod
    def create_node(self, node_data: NodeData) -> bool:
        """Create a node in the graph database."""
        pass
    
    @abstractmethod
    def create_edge(self, edge_data: EdgeData) -> bool:
        """Create an edge in the graph database."""
        pass
    
    @abstractmethod
    def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute a Cypher query."""
        pass
    
    @abstractmethod
    def execute_gremlin(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute a Gremlin query."""
        pass
    
    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[NodeData]:
        """Get a node by its ID."""
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: str, max_depth: int = 1) -> List[NodeData]:
        """Get neighbors of a node."""
        pass


class Neo4jInterface(GraphDatabaseInterface):
    """Neo4j graph database interface."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.username = username
        self.password = password
        self.graph = None
        
    def connect(self) -> bool:
        """Connect to Neo4j database."""
        if Neo4jGraph is None:
            return False
        self.graph = Neo4jGraph(self.uri, auth=(self.username, self.password))
        # Test connection
        self.graph.run("RETURN 1")
        logger.info("Connected to Neo4j successfully")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from Neo4j database."""
        if self.graph:
            # py2neo Graph doesn't have a close method
            self.graph = None
            logger.info("Disconnected from Neo4j")
    
    def create_node(self, node_data: NodeData) -> bool:
        """Create a node in Neo4j."""
        try:
            # Sanitize label name (remove spaces and special characters)
            sanitized_label = node_data.label.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            # Remove any remaining special characters
            sanitized_label = ''.join(c for c in sanitized_label if c.isalnum() or c == '_')
            
            # Filter properties to only include simple types that Neo4j can handle
            simple_properties = {}
            for key, value in node_data.properties.items():
                if isinstance(value, (str, int, float, bool)):
                    simple_properties[key] = value
                elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                    simple_properties[key] = value
                else:
                    # Convert complex types to strings
                    simple_properties[key] = str(value)
            
            cypher = f"""
            CREATE (n:{sanitized_label} {{id: $id}})
            SET n += $properties
            """
            
            self.graph.run(cypher, {
                'id': node_data.id,
                'properties': simple_properties
            })
            
            logger.info(f"Created node: {node_data.id} ({sanitized_label})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create node {node_data.id}: {e}")
            return False
    
    def create_edge(self, edge_data: EdgeData) -> bool:
        """Create an edge in Neo4j."""
        try:
            # Sanitize relationship type name (remove spaces and special characters)
            sanitized_rel_type = edge_data.relationship_type.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            # Remove any remaining special characters and ensure it starts with a letter
            sanitized_rel_type = ''.join(c for c in sanitized_rel_type if c.isalnum() or c == '_')
            # Ensure it starts with a letter (Neo4j requirement)
            if sanitized_rel_type and not sanitized_rel_type[0].isalpha():
                sanitized_rel_type = 'REL_' + sanitized_rel_type
            
            # Check if both nodes exist before creating edge
            check_cypher = """
            MATCH (a {id: $source_id}), (b {id: $target_id})
            RETURN count(a) as source_count, count(b) as target_count
            """
            
            result = self.graph.run(check_cypher, {
                'source_id': edge_data.source_id,
                'target_id': edge_data.target_id
            }).data()
            
            if result and result[0]['source_count'] > 0 and result[0]['target_count'] > 0:
                # Both nodes exist, create the edge
                cypher = f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                CREATE (a)-[r:{sanitized_rel_type}]->(b)
                SET r += $properties
                """
                
                self.graph.run(cypher, {
                    'source_id': edge_data.source_id,
                    'target_id': edge_data.target_id,
                    'properties': edge_data.properties
                })
                
                logger.info(f"Created edge: {edge_data.source_id} -[{sanitized_rel_type}]-> {edge_data.target_id}")
                return True
            else:
                logger.warning(f"Skipping edge {edge_data.source_id} -> {edge_data.target_id} (missing nodes)")
                return False
            
        except Exception as e:
            logger.error(f"Failed to create edge {edge_data.source_id} -> {edge_data.target_id}: {e}")
            return False
    
    def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute a Cypher query in Neo4j with reconnect-and-retry on connection errors."""
        import time
        start_time = time.time()

        def _run_once() -> Any:
            return self.graph.run(query, parameters or {})

        tries = 0
        last_error: Optional[Exception] = None
        while tries < 3:
            try:
                result = _run_once()
                nodes: List[NodeData] = []
                edges: List[EdgeData] = []
                for record in result:
                    for _, value in record.items():
                        if hasattr(value, 'labels'):
                            nodes.append(NodeData(
                                id=value['id'],
                                label=list(value.labels)[0],
                                properties=dict(value)
                            ))
                        elif hasattr(value, 'type'):
                            edges.append(EdgeData(
                                source_id=value.start_node['id'],
                                target_id=value.end_node['id'],
                                relationship_type=value.type,
                                properties=dict(value)
                            ))
                execution_time = time.time() - start_time
                return QueryResult(nodes=nodes, edges=edges, execution_time=execution_time, metadata={'database': 'Neo4j', 'query_type': 'cypher'})
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if 'connection has been closed' in msg or 'broken pipe' in msg or 'service unavailable' in msg:
                    # Reconnect and retry
                    try:
                        logger.warning("Neo4j connection lost; reconnecting and retrying...")
                        self.disconnect()
                        time.sleep(0.2)
                        self.connect()
                    except Exception as re:
                        last_error = re
                    tries += 1
                    continue
                else:
                    break
        logger.error(f"Cypher query failed: {last_error}")
        return QueryResult(nodes=[], edges=[], execution_time=0, metadata={'error': str(last_error)})
    
    def execute_gremlin(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute a Gremlin query in Neo4j (not directly supported)."""
        logger.warning("Gremlin queries not directly supported in Neo4j interface")
        return QueryResult(nodes=[], edges=[], execution_time=0, metadata={'error': 'Gremlin not supported'})
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeData]:
        """Get a node by its ID from Neo4j."""
        try:
            cypher = "MATCH (n) WHERE n.id = $id RETURN n"
            result = self.graph.run(cypher, {'id': node_id}).data()
            
            if result:
                node = result[0]['n']
                return NodeData(
                    id=node['id'],
                    label=list(node.labels)[0],
                    properties=dict(node)
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def get_neighbors(self, node_id: str, max_depth: int = 1) -> List[NodeData]:
        """Get neighbors of a node from Neo4j."""
        try:
            cypher = f"""
            MATCH (n)-[r*1..{max_depth}]-(neighbor)
            WHERE n.id = $id
            RETURN DISTINCT neighbor
            """
            
            result = self.graph.run(cypher, {'id': node_id}).data()
            
            neighbors = []
            for record in result:
                neighbor = record['neighbor']
                neighbors.append(NodeData(
                    id=neighbor['id'],
                    label=list(neighbor.labels)[0],
                    properties=dict(neighbor)
                ))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get neighbors for {node_id}: {e}")
            return []


class NeptuneInterface(GraphDatabaseInterface):
    """AWS Neptune graph database interface."""
    
    def __init__(self, endpoint: str, port: int = 8182):
        self.endpoint = endpoint
        self.port = port
        self.client = None
        
    def connect(self) -> bool:
        """Connect to Neptune database."""
        if client is None:
            return False
        self.client = client.Client(
            f'wss://{self.endpoint}:{self.port}/gremlin',
            'g'
        )
        # Test connection
        result = self.client.submit("g.V().limit(1)")
        result.all().result()
        logger.info("Connected to Neptune successfully")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from Neptune database."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from Neptune")
    
    def create_node(self, node_data: NodeData) -> bool:
        """Create a node in Neptune."""
        try:
            gremlin = f"g.addV('{node_data.label}').property('id', '{node_data.id}')"
            
            for key, value in node_data.properties.items():
                gremlin += f".property('{key}', {json.dumps(value)})"
            
            self.client.submit(gremlin).all().result()
            logger.info(f"Created node: {node_data.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create node {node_data.id}: {e}")
            return False
    
    def create_edge(self, edge_data: EdgeData) -> bool:
        """Create an edge in Neptune."""
        try:
            gremlin = f"""
            g.V().has('id', '{edge_data.source_id}').addE('{edge_data.relationship_type}')
            .to(g.V().has('id', '{edge_data.target_id}'))
            """
            
            for key, value in edge_data.properties.items():
                gremlin += f".property('{key}', {json.dumps(value)})"
            
            self.client.submit(gremlin).all().result()
            logger.info(f"Created edge: {edge_data.source_id} -> {edge_data.target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create edge {edge_data.source_id} -> {edge_data.target_id}: {e}")
            return False
    
    def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute a Cypher query in Neptune (not directly supported)."""
        logger.warning("Cypher queries not directly supported in Neptune interface")
        return QueryResult(nodes=[], edges=[], execution_time=0, metadata={'error': 'Cypher not supported'})
    
    def execute_gremlin(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute a Gremlin query in Neptune."""
        import time
        start_time = time.time()
        
        try:
            result = self.client.submit(query).all().result()
            nodes = []
            edges = []
            
            # Parse Gremlin results (simplified)
            for item in result:
                if hasattr(item, 'properties'):
                    # This is a simplified parsing - real implementation would be more complex
                    nodes.append(NodeData(
                        id=str(item.id),
                        label=item.label,
                        properties={}
                    ))
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                nodes=nodes,
                edges=edges,
                execution_time=execution_time,
                metadata={'database': 'Neptune', 'query_type': 'gremlin'}
            )
            
        except Exception as e:
            logger.error(f"Gremlin query failed: {e}")
            return QueryResult(nodes=[], edges=[], execution_time=0, metadata={'error': str(e)})
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeData]:
        """Get a node by its ID from Neptune."""
        try:
            gremlin = f"g.V().has('id', '{node_id}')"
            result = self.client.submit(gremlin).all().result()
            
            if result:
                node = result[0]
                return NodeData(
                    id=str(node.id),
                    label=node.label,
                    properties={}
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def get_neighbors(self, node_id: str, max_depth: int = 1) -> List[NodeData]:
        """Get neighbors of a node from Neptune."""
        try:
            gremlin = f"g.V().has('id', '{node_id}').repeat(both()).times({max_depth}).dedup()"
            result = self.client.submit(gremlin).all().result()
            
            neighbors = []
            for node in result:
                neighbors.append(NodeData(
                    id=str(node.id),
                    label=node.label,
                    properties={}
                ))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get neighbors for {node_id}: {e}")
            return []


class UnifiedGraphInterface:
    """Unified interface that can work with multiple graph database backends."""
    
    def __init__(self, backend: str = "neo4j", **config):
        self.backend = backend.lower()
        self.interface = None
        
        if self.backend == "neo4j":
            self.interface = Neo4jInterface(**config)
        elif self.backend == "neptune":
            self.interface = NeptuneInterface(**config)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def connect(self) -> bool:
        """Connect to the graph database."""
        return self.interface.connect()
    
    def disconnect(self) -> None:
        """Disconnect from the graph database."""
        self.interface.disconnect()
    
    def create_node(self, node_data: NodeData) -> bool:
        """Create a node in the graph database."""
        return self.interface.create_node(node_data)
    
    def create_edge(self, edge_data: EdgeData) -> bool:
        """Create an edge in the graph database."""
        return self.interface.create_edge(edge_data)
    
    def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute a Cypher query."""
        return self.interface.execute_cypher(query, parameters)
    
    def execute_gremlin(self, query: str, parameters: Optional[Dict] = None) -> QueryResult:
        """Execute a Gremlin query."""
        return self.interface.execute_gremlin(query, parameters)
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeData]:
        """Get a node by its ID."""
        return self.interface.get_node_by_id(node_id)
    
    def get_neighbors(self, node_id: str, max_depth: int = 1) -> List[NodeData]:
        """Get neighbors of a node."""
        return self.interface.get_neighbors(node_id, max_depth)


if __name__ == "__main__":
    # Test unified interface
    try:
        # Test Neo4j interface (would require actual Neo4j instance)
        # neo4j_interface = UnifiedGraphInterface("neo4j", uri="bolt://localhost:7687")
        # print("Neo4j interface created successfully")
        
        # Test Neptune interface (would require actual Neptune instance)
        # neptune_interface = UnifiedGraphInterface("neptune", endpoint="your-neptune-endpoint")
        # print("Neptune interface created successfully")
        
        print("Graph database interface modules loaded successfully")
        
    except Exception as e:
        print(f"Error testing interfaces: {e}")
