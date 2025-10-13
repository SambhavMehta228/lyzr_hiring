"""
Main pipeline for Document-to-Graph processing and Agentic Retrieval.
Orchestrates the entire system from data loading to query execution.
"""

import logging
from typing import Dict, List, Any, Optional
import time
import os
from dataclasses import dataclass
from dotenv import load_dotenv

from .data_loader import DBpediaDataLoader
from .ontology_generator import OntologyGenerator, Entity, Relationship, OntologySchema
from .graph_builder import KnowledgeGraphBuilder
from .agentic_retrieval import AgenticRetrievalSystem
from .graph_db_interface import UnifiedGraphInterface, NodeData, EdgeData

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    dataset_name: str = "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K"
    openai_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    graph_backend: str = "neo4j"  # or "neptune"
    graph_config: Dict[str, Any] = None
    batch_size: int = 100
    max_entities: int = 1000  # Limit for demo purposes
    enable_entity_resolution: bool = True
    enable_relationship_extraction: bool = True
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables."""
        return cls(
            dataset_name=os.getenv("DATASET_NAME", "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            graph_backend=os.getenv("GRAPH_BACKEND", "neo4j"),
            graph_config={
                "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password"),
                "endpoint": os.getenv("NEPTUNE_ENDPOINT", ""),
                "port": int(os.getenv("NEPTUNE_PORT", "8182"))
            },
            batch_size=int(os.getenv("BATCH_SIZE", "50")),
            max_entities=int(os.getenv("MAX_ENTITIES", "1000")),
            enable_entity_resolution=os.getenv("ENABLE_ENTITY_RESOLUTION", "true").lower() == "true",
            enable_relationship_extraction=os.getenv("ENABLE_RELATIONSHIP_EXTRACTION", "true").lower() == "true"
        )


@dataclass
class PipelineStats:
    """Statistics about pipeline execution."""
    total_entities_processed: int = 0
    total_relationships_extracted: int = 0
    entities_resolved: int = 0
    graph_nodes_created: int = 0
    graph_edges_created: int = 0
    execution_time: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DocumentToGraphPipeline:
    """Main pipeline for processing documents into knowledge graphs."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = None
        self.ontology_generator = None
        self.graph_builder = None
        self.graph_interface = None
        self.stats = PipelineStats()
        
    def initialize(self) -> bool:
        """Initialize all pipeline components."""
        try:
            # Initialize data loader
            self.data_loader = DBpediaDataLoader(self.config.dataset_name)
            
            # Initialize ontology generator
            if not self.config.openai_api_key:
                logger.warning("No OpenAI API key provided - ontology generation will be limited")
            self.ontology_generator = OntologyGenerator(self.config.openai_api_key)
            
            # Initialize graph builder
            self.graph_builder = KnowledgeGraphBuilder(self.config.qdrant_url, self.config.qdrant_api_key)
            
            # Initialize graph database interface
            graph_config = self.config.graph_config or {}
            if self.config.graph_backend == "neo4j":
                self.graph_interface = UnifiedGraphInterface(
                    "neo4j",
                    uri=graph_config.get("uri", "bolt://localhost:7687"),
                    username=graph_config.get("username", "neo4j"),
                    password=graph_config.get("password", "password")
                )
            elif self.config.graph_backend == "neptune":
                self.graph_interface = UnifiedGraphInterface(
                    "neptune",
                    endpoint=graph_config.get("endpoint", ""),
                    port=graph_config.get("port", 8182)
                )
            else:
                self.graph_interface = None
            
            logger.info("Pipeline components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.stats.errors.append(f"Initialization error: {e}")
            return False
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Entity]:
        """Process a batch of entities."""
        entities = []
        
        for entity_data in batch_data:
            try:
                # Create entity using ontology generator
                entity = self.ontology_generator.create_entity(entity_data)
                entities.append(entity)
                
            except Exception as e:
                logger.error(f"Failed to process entity {entity_data.get('id', 'unknown')}: {e}")
                self.stats.errors.append(f"Entity processing error: {e}")
        
        return entities
    
    def extract_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        
        if not self.config.enable_relationship_extraction:
            return relationships
        
        for entity in entities:
            try:
                # Extract relationships for each entity, passing existing entities for better matching
                entity_relationships = self.ontology_generator.extract_relationships({
                    '_id': entity.id,
                    'title': entity.name,
                    'text': entity.description
                }, existing_entities=entities)
                relationships.extend(entity_relationships)
                
            except Exception as e:
                logger.error(f"Failed to extract relationships for {entity.id}: {e}")
                self.stats.errors.append(f"Relationship extraction error: {e}")
        
        return relationships
    
    def resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve and deduplicate entities."""
        if not self.config.enable_entity_resolution:
            return entities
        
        try:
            resolved_entities = self.ontology_generator.resolve_entities(entities)
            self.stats.entities_resolved = len(entities) - len(resolved_entities)
            return resolved_entities
            
        except Exception as e:
            logger.error(f"Failed to resolve entities: {e}")
            self.stats.errors.append(f"Entity resolution error: {e}")
            return entities
    
    def build_knowledge_graph(self, entities: List[Entity], relationships: List[Relationship]) -> None:
        """Build the knowledge graph from entities and relationships."""
        try:
            # Build in-memory graph
            self.graph_builder.build_from_entities(entities, relationships)
            self.stats.graph_nodes_created = len(entities)
            self.stats.graph_edges_created = len(relationships)
            
            # Optionally sync to persistent graph database
            if self.graph_interface:
                self._sync_to_graph_db(entities, relationships)
            
            logger.info(f"Knowledge graph built with {len(entities)} nodes and {len(relationships)} edges")
            
        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {e}")
            self.stats.errors.append(f"Graph building error: {e}")
    
    def _sync_to_graph_db(self, entities: List[Entity], relationships: List[Relationship]) -> None:
        """Sync the knowledge graph to persistent database."""
        try:
            # Connect to graph database
            if not self.graph_interface.connect():
                logger.warning("Failed to connect to graph database - continuing with in-memory graph")
                return

            # Prefer batched UNWIND writes for Neo4j to avoid connection contention
            if hasattr(self.graph_interface, 'backend') and self.graph_interface.backend == 'neo4j':
                # Group entities by label for batched node upserts
                from collections import defaultdict
                label_to_rows = defaultdict(list)
                for entity in entities:
                    label_to_rows[entity.type].append({
                        'id': entity.id,
                        'properties': {
                            'name': entity.name,
                            'description': entity.description,
                            **entity.properties
                        }
                    })

                batch_size = 200
                total_nodes = 0
                for label, rows in label_to_rows.items():
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i+batch_size]
                        cypher = f"""
                        UNWIND $rows AS row
                        MERGE (n:{label} {{ id: row.id }})
                        SET n += row.properties
                        """
                        self.graph_interface.execute_cypher(cypher, { 'rows': batch })
                        total_nodes += len(batch)

                # Group relationships by type and write in batches
                reltype_to_rows = defaultdict(list)
                valid_ids = {e.id for e in entities}
                for rel in relationships:
                    if rel.source_id in valid_ids and rel.target_id in valid_ids:
                        reltype_to_rows[rel.relationship_type].append({
                            'source_id': rel.source_id,
                            'target_id': rel.target_id,
                            'properties': rel.properties
                        })

                successful_edges_inner = 0
                for rel_type, rows in reltype_to_rows.items():
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i+batch_size]
                        cypher = f"""
                        UNWIND $rows AS row
                        MATCH (a { {id: row.source_id} }), (b { {id: row.target_id} })
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += row.properties
                        """
                        # Fix curly braces spacing for Cypher parameter maps
                        cypher = cypher.replace('{ {', '{').replace('} }', '}')
                        self.graph_interface.execute_cypher(cypher, { 'rows': batch })
                        successful_edges_inner += len(batch)

                logger.info(f"Knowledge graph synced: {total_nodes} nodes, {successful_edges_inner} edges")
            else:
                # Fallback: sequential creates (thread-safe for single connection backends)
                for entity in entities:
                    node_data = NodeData(
                        id=entity.id,
                        label=entity.type,
                        properties={
                            'name': entity.name,
                            'description': entity.description,
                            **entity.properties
                        }
                    )
                    self.graph_interface.create_node(node_data)

                successful_edges = 0
                entity_ids = {e.id for e in entities}
                for relationship in relationships:
                    if relationship.source_id in entity_ids and relationship.target_id in entity_ids:
                        edge_data = EdgeData(
                            source_id=relationship.source_id,
                            target_id=relationship.target_id,
                            relationship_type=relationship.relationship_type,
                            properties=relationship.properties
                        )
                        if self.graph_interface.create_edge(edge_data):
                            successful_edges += 1
                logger.info(f"Knowledge graph synced: {len(entities)} nodes, {successful_edges} edges")
            
        except Exception as e:
            logger.error(f"Failed to sync to graph database: {e}")
            self.stats.errors.append(f"Graph sync error: {e}")
        finally:
            if self.graph_interface:
                self.graph_interface.disconnect()
    
    def run_pipeline(self) -> PipelineStats:
        """Run the complete document-to-graph pipeline."""
        start_time = time.time()
        
        try:
            # Initialize pipeline
            if not self.initialize():
                return self.stats
            
            # Load dataset
            logger.info("Loading dataset...")
            self.data_loader.load_dataset()
            
            # Process entities in batches
            total_processed = 0
            all_entities = []
            all_relationships = []
            
            for batch in self.data_loader.iterate_entities(self.config.batch_size):
                if total_processed >= self.config.max_entities:
                    break
                
                logger.info(f"Processing batch {total_processed // self.config.batch_size + 1}...")
                
                # Process batch
                batch_entities = self.process_batch(batch)
                all_entities.extend(batch_entities)
                
                # Extract relationships
                batch_relationships = self.extract_relationships(batch_entities)
                all_relationships.extend(batch_relationships)
                
                total_processed += len(batch)
                self.stats.total_entities_processed = total_processed
                self.stats.total_relationships_extracted = len(all_relationships)
                
                logger.info(f"Processed {total_processed} entities, {len(all_relationships)} relationships")
            
            # Resolve entities
            logger.info("Resolving entities...")
            resolved_entities = self.resolve_entities(all_entities)
            
            # Build knowledge graph
            logger.info("Building knowledge graph...")
            self.build_knowledge_graph(resolved_entities, all_relationships)
            
            self.stats.execution_time = time.time() - start_time
            logger.info(f"Pipeline completed in {self.stats.execution_time:.2f} seconds")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.stats.errors.append(f"Pipeline execution error: {e}")
            self.stats.execution_time = time.time() - start_time
            return self.stats
    
    def get_ontology_schema(self) -> Optional[OntologySchema]:
        """Get the generated ontology schema."""
        if not self.graph_builder or not self.graph_builder.graph.nodes:
            return None
        
        # Extract entities from graph for schema generation
        entities = []
        for node_id, data in self.graph_builder.graph.nodes(data=True):
            entity = Entity(
                id=data.get('original_id', node_id),
                name=data.get('name', ''),
                type=data.get('label', ''),
                description=data.get('description', ''),
                properties=data
            )
            entities.append(entity)
        
        return self.ontology_generator.build_ontology_schema(entities)


class AgenticGraphRAGService:
    """Complete service combining document-to-graph pipeline with agentic retrieval."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline = DocumentToGraphPipeline(config)
        self.retrieval_system = None
        
    def initialize(self) -> bool:
        """Initialize the complete service."""
        if not self.pipeline.initialize():
            return False
        
        # Initialize retrieval system
        self.retrieval_system = AgenticRetrievalSystem(
            self.pipeline.graph_builder,
            self.config.openai_api_key
        )
        
        logger.info("Agentic Graph RAG Service initialized successfully")
        return True
    
    def build_knowledge_base(self) -> PipelineStats:
        """Build the knowledge base from documents."""
        return self.pipeline.run_pipeline()
    
    def query(self, query: str, limit: int = 10, use_multi_step: bool = False) -> Dict[str, Any]:
        """Execute a query against the knowledge base."""
        if not self.retrieval_system:
            return {"error": "Service not initialized"}
        
        try:
            if use_multi_step:
                response = self.retrieval_system.multi_step_reasoning(query, max_steps=3)
            else:
                response = self.retrieval_system.execute_query(query, limit)
            
            return {
                "query": response.query,
                "results": [
                    {
                        "id": result.entity_id,
                        "name": result.name,
                        "type": result.entity_type,
                        "description": result.description,
                        "score": result.score,
                        "reasoning": result.reasoning
                    }
                    for result in response.results
                ],
                "reasoning_chain": response.reasoning_chain,
                "search_strategy": response.search_strategy,
                "confidence": response.confidence,
                "execution_time": response.execution_time
            }
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"error": str(e)}
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        stats = self.pipeline.stats
        
        graph_stats = {}
        if self.pipeline.graph_builder:
            graph_stats = self.pipeline.graph_builder.get_graph_statistics()
        
        return {
            "pipeline_stats": {
                "entities_processed": stats.total_entities_processed,
                "relationships_extracted": stats.total_relationships_extracted,
                "entities_resolved": stats.entities_resolved,
                "execution_time": stats.execution_time,
                "errors": stats.errors
            },
            "graph_stats": graph_stats,
            "config": {
                "dataset": self.config.dataset_name,
                "backend": self.config.graph_backend,
                "batch_size": self.config.batch_size,
                "max_entities": self.config.max_entities
            }
        }


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = PipelineConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        max_entities=100,  # Small sample for demo
        batch_size=10
    )
    
    service = AgenticGraphRAGService(config)
    
    if service.initialize():
        print("Service initialized successfully")
        
        # Build knowledge base
        stats = service.build_knowledge_base()
        print(f"Knowledge base built: {stats.total_entities_processed} entities processed")
        
        # Example query
        result = service.query("Find scientists who worked at universities")
        print(f"Query result: {len(result.get('results', []))} entities found")
    else:
        print("Service initialization failed")
