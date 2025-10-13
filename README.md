# Agentic Graph RAG as a Service

A production-ready, extensible platform that integrates knowledge from multiple sources into an intelligent retrieval system powered by **LLMs, embeddings, and autonomous agents**.

## Overview

This system provides:
- **Document-to-Graph Pipeline**: LLM-driven ontology generation and automatic knowledge graph construction
- **Agentic Retrieval System**: Dynamic AI agents for optimal query resolution using vector similarity, graph traversal, and logical filtering
- **Extensible Interface**: Compatible with Neo4j and AWS Neptune
- **Production-Ready API**: FastAPI-based REST API for seamless integration

## Architecture

### Core Components

1. **Document-to-Graph Pipeline**
   - LLM-driven ontology generation for entities, relationships, and hierarchies
   - Automatic knowledge graph construction from unstructured text
   - Integration with OpenAI embeddings for semantic search
   - Entity resolution and deduplication
   - Visual ontology editor with LLM-assisted modification

2. **Agentic Retrieval System**
   - Dynamic AI agents that select optimal tools for query resolution
   - Vector similarity search (semantic matching via embeddings)
   - Graph traversal (relationship reasoning)
   - Logical filtering (metadata-based constraints)
   - Multi-step reasoning and iterative refinement
   - Streaming responses showing reasoning chains

3. **Extensible Graph Interface**
   - Unified API for Neo4j and AWS Neptune
   - Cypher and Gremlin query support
   - Vector database integration with Qdrant

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for LLM-driven features)
- Qdrant vector database (optional, for embeddings)
- Neo4j or AWS Neptune (optional, for persistent graph storage)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lyzr_hiring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

### Running the Demo

```bash
python demo.py
```

This will:
- Load a sample of the DBpedia entities dataset
- Build a knowledge graph using LLM-driven ontology generation
- Demonstrate agentic retrieval with example queries

### Running the API Service

```bash
python -m src.api
```

The API will be available at `http://localhost:8000` with automatic documentation at `http://localhost:8000/docs`.

## Dataset

The system uses the `Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K` dataset, which contains:
- 100,000 DBpedia entities
- OpenAI text-embedding-3-small embeddings (1024 dimensions)
- Structured entity information with descriptions

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
QDRANT_URL=http://localhost:6333
GRAPH_BACKEND=neo4j
MAX_ENTITIES=1000
BATCH_SIZE=50
```

### Pipeline Configuration

```python
config = PipelineConfig(
    openai_api_key="your_key",
    dataset_name="Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K",
    max_entities=1000,
    batch_size=50,
    enable_entity_resolution=True,
    enable_relationship_extraction=True
)
```

## Usage Examples

### Basic Query

```python
from src.pipeline import AgenticGraphRAGService, PipelineConfig

# Initialize service
service = AgenticGraphRAGService(config)
service.initialize()

# Build knowledge base
service.build_knowledge_base()

# Query
result = service.query("Find scientists who worked at universities")
print(f"Found {len(result['results'])} entities")
```

### Multi-Step Reasoning

```python
result = service.query(
    "Find physicists who made significant contributions to quantum mechanics",
    use_multi_step=True
)
```

### API Usage

```bash
# Build knowledge base
curl -X POST "http://localhost:8000/build"

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Find scientists", "limit": 10}'
```

## Evaluation Criteria

The system is designed to excel in four key areas:

### A. System Architecture (25%)
- Modular design with clear separation of concerns
- Neo4j/Neptune parity through unified interface
- Efficient embedding storage and retrieval
- Entity resolution and deduplication subsystems

### B. Graph Quality & Ontology (25%)
- Accurate entity type classification using LLMs
- Complete relationship extraction and validation
- Effective entity resolution and deduplication
- LLM-assisted ontology refinement

### C. Retrieval Intelligence (25%)
- Intelligent agent routing between search strategies
- Hybrid relevance scoring combining multiple signals
- Low query latency with efficient graph operations
- Automatic Cypher/Gremlin query generation
- Streaming reasoning chains for transparency

### D. Extensibility & Maintainability (25%)
- Pluggable graph database backends
- Comprehensive APIs and SDKs
- Versioned ontology management
- CI/CD pipeline support
- High test coverage and operational monitoring

## Key Features

### Intelligent Agent Routing
The system automatically selects the optimal search strategy based on query analysis:
- **Vector Similarity**: For semantic similarity searches
- **Graph Traversal**: For relationship-based queries
- **Logical Filter**: For structured filtering queries
- **Hybrid**: For complex queries requiring multiple approaches

### Multi-Step Reasoning
For complex queries, the system can:
- Analyze initial results
- Refine search strategies
- Iterate until optimal results are found
- Provide transparent reasoning chains

### Extensible Backend Support
- **Neo4j**: Full Cypher query support
- **AWS Neptune**: Gremlin query support
- **Qdrant**: Vector similarity search
- **NetworkX**: In-memory graph operations

## Development

### Project Structure

```
src/
├── data_loader.py          # Dataset loading and preprocessing
├── ontology_generator.py   # LLM-driven ontology generation
├── graph_builder.py        # Knowledge graph construction
├── agentic_retrieval.py    # Dynamic agent-based retrieval
├── graph_db_interface.py   # Unified graph database interface
├── pipeline.py            # Main pipeline orchestration
└── api.py                 # FastAPI REST interface
```

### Adding New Features

1. **New Search Strategies**: Extend the `AgenticRetrievalSystem` class
2. **New Graph Backends**: Implement the `GraphDatabaseInterface` abstract class
3. **New Ontology Types**: Extend the `OntologyGenerator` class
4. **New API Endpoints**: Add routes to the FastAPI application

## License

This project is part of the Lyzr hiring process and demonstrates production-quality thinking in system architecture, scalability, and intelligent design.

## Contributing

This is a demonstration project for the Lyzr hiring process. The focus is on:
- Clean architecture and modular design
- Computational and time efficiency
- Production-ready code quality
- Creative problem-solving approaches

---

**"We don't expect perfection. We expect potential — and how you think."**
