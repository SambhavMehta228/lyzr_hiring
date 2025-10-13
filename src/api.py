"""
FastAPI interface for the Agentic Graph RAG Service.
Provides REST API endpoints for querying and managing the knowledge graph.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import uvicorn
from contextlib import asynccontextmanager
import asyncio

from .pipeline import AgenticGraphRAGService, PipelineConfig, PipelineStats

logger = logging.getLogger(__name__)

# Global service instance
service_instance: Optional[AgenticGraphRAGService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global service_instance
    
    # Startup
    logger.info("Starting Agentic Graph RAG Service...")
    
    # Initialize service (without building knowledge base yet)
    config = PipelineConfig(
        openai_api_key="",  # Will be set via API
        max_entities=1000,
        batch_size=50
    )
    service_instance = AgenticGraphRAGService(config)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agentic Graph RAG Service...")


# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    limit: int = 10
    use_multi_step: bool = False


class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    reasoning_chain: List[str]
    search_strategy: str
    confidence: float
    execution_time: float


class ServiceConfig(BaseModel):
    openai_api_key: str
    dataset_name: str = "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    graph_backend: str = "neo4j"
    max_entities: int = 1000
    batch_size: int = 50
    enable_entity_resolution: bool = True
    enable_relationship_extraction: bool = True


class ServiceStats(BaseModel):
    pipeline_stats: Dict[str, Any]
    graph_stats: Dict[str, Any]
    config: Dict[str, Any]


class BuildStatus(BaseModel):
    status: str  # "building", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    stats: Optional[PipelineStats] = None


# FastAPI app
app = FastAPI(
    title="Agentic Graph RAG as a Service",
    description="Production-ready platform for intelligent retrieval using LLMs, embeddings, and autonomous agents",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global build status
build_status = BuildStatus(status="idle", progress=0.0, message="Service ready")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Agentic Graph RAG as a Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "build": "/build",
            "stats": "/stats",
            "config": "/config",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global service_instance
    
    if service_instance is None:
        return {"status": "unhealthy", "message": "Service not initialized"}
    
    try:
        # Check if knowledge graph is built
        stats = service_instance.get_service_stats()
        graph_nodes = stats.get("graph_stats", {}).get("total_nodes", 0)
        
        return {
            "status": "healthy" if graph_nodes > 0 else "building",
            "graph_nodes": graph_nodes,
            "service_initialized": True
        }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


@app.post("/config")
async def configure_service(config: ServiceConfig):
    """Configure the service with API keys and parameters."""
    global service_instance
    
    if service_instance is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    try:
        # Update service configuration
        pipeline_config = PipelineConfig(
            openai_api_key=config.openai_api_key,
            dataset_name=config.dataset_name,
            qdrant_url=config.qdrant_url,
            qdrant_api_key=config.qdrant_api_key or "",
            graph_backend=config.graph_backend,
            max_entities=config.max_entities,
            batch_size=config.batch_size,
            enable_entity_resolution=config.enable_entity_resolution,
            enable_relationship_extraction=config.enable_relationship_extraction
        )
        
        # Reinitialize service with new config
        service_instance = AgenticGraphRAGService(pipeline_config)
        
        if service_instance.initialize():
            return {"message": "Service configured successfully", "config": config.dict()}
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize service with new configuration")
            
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build")
async def build_knowledge_base(background_tasks: BackgroundTasks):
    """Build the knowledge base from documents."""
    global service_instance, build_status
    
    if service_instance is None:
        raise HTTPException(status_code=500, detail="Service not configured")
    
    if build_status.status == "building":
        raise HTTPException(status_code=409, detail="Knowledge base is already being built")
    
    # Start background build task
    background_tasks.add_task(build_knowledge_base_task)
    
    build_status = BuildStatus(status="building", progress=0.0, message="Starting knowledge base build...")
    
    return {"message": "Knowledge base build started", "status": "building"}


async def build_knowledge_base_task():
    """Background task to build the knowledge base."""
    global service_instance, build_status
    
    try:
        build_status.progress = 0.1
        build_status.message = "Loading dataset..."
        
        # Build knowledge base without blocking the event loop
        stats = await asyncio.to_thread(service_instance.build_knowledge_base)
        
        build_status.status = "completed"
        build_status.progress = 1.0
        build_status.message = f"Knowledge base built successfully with {stats.total_entities_processed} entities"
        build_status.stats = stats
        
        logger.info(f"Knowledge base build completed: {stats.total_entities_processed} entities processed")
        
    except Exception as e:
        logger.error(f"Knowledge base build failed: {e}")
        build_status.status = "failed"
        build_status.progress = 0.0
        build_status.message = f"Build failed: {str(e)}"


@app.get("/build/status")
async def get_build_status():
    """Get the current build status."""
    global build_status
    return build_status


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base using agentic retrieval."""
    global service_instance
    
    if service_instance is None:
        raise HTTPException(status_code=500, detail="Service not configured")
    
    if build_status.status != "completed":
        raise HTTPException(status_code=409, detail="Knowledge base not ready. Please build it first.")
    
    try:
        # Execute query without blocking the event loop
        result = await asyncio.to_thread(
            service_instance.query,
            request.query,
            request.limit,
            request.use_multi_step
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=ServiceStats)
async def get_service_stats():
    """Get comprehensive service statistics."""
    global service_instance
    
    if service_instance is None:
        raise HTTPException(status_code=500, detail="Service not configured")
    
    try:
        stats = service_instance.get_service_stats()
        return ServiceStats(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get service stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ontology")
async def get_ontology_schema():
    """Get the generated ontology schema."""
    global service_instance
    
    if service_instance is None:
        raise HTTPException(status_code=500, detail="Service not configured")
    
    try:
        schema = service_instance.pipeline.get_ontology_schema()
        if schema is None:
            raise HTTPException(status_code=404, detail="Ontology schema not available")
        
        return {
            "entity_types": schema.entity_types,
            "relationship_types": schema.relationship_types,
            "hierarchies": schema.hierarchies
        }
        
    except Exception as e:
        logger.error(f"Failed to get ontology schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/export/cypher")
async def export_cypher_queries(limit: Optional[int] = None):
    """Export the knowledge graph as Cypher queries."""
    global service_instance
    
    if service_instance is None:
        raise HTTPException(status_code=500, detail="Service not configured")
    
    if build_status.status != "completed":
        raise HTTPException(status_code=409, detail="Knowledge base not ready")
    
    try:
        cypher_queries = service_instance.pipeline.graph_builder.export_to_cypher(limit)
        return {
            "queries": cypher_queries,
            "total_queries": len(cypher_queries),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to export Cypher queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/search/similar")
async def search_similar_entities(query_embedding: List[float], limit: int = 10):
    """Search for similar entities using vector similarity."""
    global service_instance
    
    if service_instance is None:
        raise HTTPException(status_code=500, detail="Service not configured")
    
    if build_status.status != "completed":
        raise HTTPException(status_code=409, detail="Knowledge base not ready")
    
    try:
        results = service_instance.pipeline.graph_builder.search_similar_entities(
            query_embedding, limit
        )
        return {
            "results": results,
            "total_results": len(results),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the API server
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
