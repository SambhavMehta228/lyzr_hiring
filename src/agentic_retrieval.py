"""
Agentic retrieval system with dynamic AI agents for query resolution.
Implements vector similarity search, graph traversal, and logical filtering.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
from openai import OpenAI
from math import log

try:
    # Lightweight lexical ranking
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

from .graph_builder import KnowledgeGraphBuilder
from .ontology_generator import Entity

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of search strategies available to agents."""
    VECTOR_SIMILARITY = "vector_similarity"
    GRAPH_TRAVERSAL = "graph_traversal"
    LOGICAL_FILTER = "logical_filter"
    HYBRID = "hybrid"
    LEXICAL = "lexical"


@dataclass
class QueryContext:
    """Context for query processing and reasoning."""
    original_query: str
    processed_query: str
    search_type: SearchType
    entities_mentioned: List[str]
    relationships_mentioned: List[str]
    filters: Dict[str, Any]
    reasoning_steps: List[str]


@dataclass
class SearchResult:
    """Result from a search operation."""
    entity_id: str
    name: str
    entity_type: str
    description: str
    score: float
    search_type: SearchType
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class AgentResponse:
    """Response from an agentic retrieval operation."""
    query: str
    results: List[SearchResult]
    reasoning_chain: List[str]
    search_strategy: str
    confidence: float
    execution_time: float


class AgenticRetrievalSystem:
    """Dynamic AI agents for intelligent query resolution."""
    
    def __init__(self, graph_builder: KnowledgeGraphBuilder, openai_api_key: str):
        self.graph_builder = graph_builder
        self.client = OpenAI(api_key=openai_api_key)
        # Initialize BM25 index from current graph if available
        self._bm25_tokenized_corpus: List[List[str]] = []
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_node_ids: List[str] = []
        self._build_bm25_index()
        # Optional cross-encoder reranker
        self._cross_encoder: Optional[CrossEncoder] = None
        self._cross_encoder_model_name: Optional[str] = None

    def enable_cross_encoder(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> bool:
        """Optionally enable a local cross-encoder reranker."""
        if CrossEncoder is None:
            logger.warning("sentence-transformers not installed; cross-encoder disabled")
            return False
        try:
            self._cross_encoder = CrossEncoder(model_name)
            self._cross_encoder_model_name = model_name
            logger.info(f"Cross-encoder enabled: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load cross-encoder {model_name}: {e}")
            self._cross_encoder = None
            return False

    def _simple_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [t for t in ''.join([c.lower() if c.isalnum() else ' ' for c in text]).split() if t]

    def _build_bm25_index(self) -> None:
        if BM25Okapi is None or self.graph_builder is None or self.graph_builder.graph is None:
            return
        tokenized_docs: List[List[str]] = []
        node_ids: List[str] = []
        for node_id, data in self.graph_builder.graph.nodes(data=True):
            text = f"{data.get('name','')} {data.get('description','')}"
            tokens = self._simple_tokenize(text)
            tokenized_docs.append(tokens)
            node_ids.append(node_id)
        if tokenized_docs:
            try:
                self._bm25_index = BM25Okapi(tokenized_docs)
                self._bm25_tokenized_corpus = tokenized_docs
                self._bm25_node_ids = node_ids
            except Exception:
                self._bm25_index = None
        
    def analyze_query(self, query: str) -> QueryContext:
        """Analyze query to determine optimal search strategy."""
        prompt = f"""
        Analyze this query and determine the best search strategy:
        1. VECTOR_SIMILARITY: For semantic similarity searches
        2. GRAPH_TRAVERSAL: For relationship-based queries
        3. LOGICAL_FILTER: For structured filtering queries
        4. HYBRID: For complex queries requiring multiple approaches
        
        Query: "{query}"
        
        Return JSON with:
        {{
            "search_type": "vector_similarity|graph_traversal|logical_filter|hybrid",
            "entities_mentioned": ["entity1", "entity2"],
            "relationships_mentioned": ["relationship1", "relationship2"],
            "filters": {{"key": "value"}},
            "reasoning": "explanation of choice"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            # Handle cases where LLM doesn't return valid JSON
            if not response_text.startswith('{'):
                # Try to extract JSON from the response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]
                else:
                    # Fallback to default analysis
                    return QueryContext(
                        original_query=query,
                        processed_query=query,
                        search_type=SearchType.VECTOR_SIMILARITY,
                        entities_mentioned=[],
                        relationships_mentioned=[],
                        filters={},
                        reasoning_steps=["Fallback to vector similarity due to invalid JSON response"]
                    )
            
            analysis = json.loads(response_text)
            
            return QueryContext(
                original_query=query,
                processed_query=query,
                search_type=SearchType(analysis['search_type']),
                entities_mentioned=analysis.get('entities_mentioned', []),
                relationships_mentioned=analysis.get('relationships_mentioned', []),
                filters=analysis.get('filters', {}),
                reasoning_steps=[analysis.get('reasoning', '')]
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            return QueryContext(
                original_query=query,
                processed_query=query,
                search_type=SearchType.VECTOR_SIMILARITY,
                entities_mentioned=[],
                relationships_mentioned=[],
                filters={},
                reasoning_steps=[f"Fallback to vector similarity due to error: {e}"]
            )
    
    def vector_similarity_search(self, query: str, context: QueryContext, limit: int = 10) -> List[SearchResult]:
        """Perform vector similarity search using embeddings."""
        # Generate query embedding
        try:
            embedding_response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                dimensions=1024  # Match the dataset embedding dimensions
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Search similar entities
            similar_entities = self.graph_builder.search_similar_entities(query_embedding, limit)
            
            results = []
            for entity in similar_entities:
                results.append(SearchResult(
                    entity_id=entity['id'],
                    name=entity['name'],
                    entity_type=entity['type'],
                    description=entity['description'],
                    score=entity['score'],
                    search_type=SearchType.VECTOR_SIMILARITY,
                    reasoning=f"Semantic similarity to query: {query[:100]}",
                    metadata={'embedding_similarity': entity['score']}
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []

    def bm25_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Perform BM25 lexical search over node names/descriptions."""
        if self._bm25_index is None:
            return []
        try:
            tokens = self._simple_tokenize(query)
            scores = self._bm25_index.get_scores(tokens)
            # Get top-k indices
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:max(limit, 20)]
            results: List[SearchResult] = []
            for idx, score in ranked:
                node_id = self._bm25_node_ids[idx]
                data = self.graph_builder.graph.nodes[node_id]
                results.append(SearchResult(
                    entity_id=node_id,
                    name=data.get('name', ''),
                    entity_type=data.get('label', ''),
                    description=data.get('description', ''),
                    score=float(score),
                    search_type=SearchType.LEXICAL,
                    reasoning="BM25 lexical relevance",
                    metadata={"bm25": float(score)}
                ))
            return results[:limit]
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _reciprocal_rank_fusion(self, lists: List[List[SearchResult]], k: int = 60, limit: int = 10) -> List[SearchResult]:
        """Fuse ranked lists with Reciprocal Rank Fusion."""
        scores: Dict[str, Tuple[SearchResult, float]] = {}
        for result_list in lists:
            for rank, item in enumerate(result_list, start=1):
                fused = 1.0 / (k + rank)
                prior = scores.get(item.entity_id)
                if prior is None or item.score > prior[0].score:
                    scores[item.entity_id] = (item, fused + (0.0 if prior is None else prior[1]))
                else:
                    scores[item.entity_id] = (prior[0], fused + prior[1])
        fused_results = [SearchResult(
            entity_id=itm.entity_id,
            name=itm.name,
            entity_type=itm.entity_type,
            description=itm.description,
            score=score_sum,
            search_type=SearchType.HYBRID,
            reasoning=f"RRF fused ({itm.search_type.value})",
            metadata={"rrf": score_sum}
        ) for itm, score_sum in scores.values()]
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:limit]

    def _heuristic_rerank(self, query: str, candidates: List[SearchResult], limit: int) -> List[SearchResult]:
        """Rerank with lightweight heuristics: name overlap + short description bias."""
        q_tokens = set(self._simple_tokenize(query))
        def score(c: SearchResult) -> float:
            name_tokens = set(self._simple_tokenize(c.name))
            overlap = len(q_tokens & name_tokens)
            desc_len = len(c.description) or 1
            bonus = 1.0 + overlap
            length_penalty = 1.0 / (1.0 + log(desc_len))
            return c.score * 0.6 + bonus * 0.3 + length_penalty * 0.1
        reranked = sorted(candidates, key=score, reverse=True)
        return reranked[:limit]

    def _cross_encoder_rerank(self, query: str, candidates: List[SearchResult], limit: int) -> List[SearchResult]:
        """Rerank using a cross-encoder if enabled; fallback to heuristic if not."""
        if not self._cross_encoder:
            return self._heuristic_rerank(query, candidates, limit)
        try:
            pairs = [(query, f"{c.name}. {c.description}") for c in candidates]
            scores = self._cross_encoder.predict(pairs)
            # Combine cross-encoder score with existing score
            combined = list(zip(candidates, scores))
            combined.sort(key=lambda x: (float(x[1]) * 0.7 + float(x[0].score) * 0.3), reverse=True)
            return [c for c, _ in combined][:limit]
        except Exception as e:
            logger.error(f"Cross-encoder rerank failed: {e}")
            return self._heuristic_rerank(query, candidates, limit)
    
    def graph_traversal_search(self, query: str, context: QueryContext, limit: int = 10) -> List[SearchResult]:
        """Perform graph traversal with confidence-weighted scoring and optional metapaths."""
        results: List[SearchResult] = []
        try:
            graph = self.graph_builder.graph
            if graph is None:
                return []

            # Derive seeds from mentioned entities or fallback to top-degree nodes matching query tokens
            seeds: List[str] = []
            q_lower = query.lower()
            for node_id, data in graph.nodes(data=True):
                if any(name.lower() in data.get('name', '').lower() for name in context.entities_mentioned):
                    seeds.append(node_id)
            if not seeds:
                for node_id, data in graph.nodes(data=True):
                    if any(tok in data.get('name', '').lower() for tok in q_lower.split()[:3]):
                        seeds.append(node_id)
                        if len(seeds) >= 3:
                            break

            # Define optional metapaths as sequences of node labels
            # Example: Person -> Organization -> Place
            metapaths: List[List[str]] = []
            if context.relationships_mentioned:
                # Heuristic: if works/worksAt mentioned, prefer Person-Organization paths
                if any('work' in r.lower() for r in context.relationships_mentioned):
                    metapaths.append(['Person', 'Organization'])

            def label_of(node_id: str) -> str:
                return graph.nodes[node_id].get('label', '')

            def path_matches_metapath(path: List[str]) -> bool:
                if not metapaths:
                    return True
                labels = [label_of(n) for n in path]
                for mp in metapaths:
                    if len(labels) >= len(mp) and labels[:len(mp)] == mp:
                        return True
                return False

            # BFS up to depth 2 with confidence weights on edges
            max_depth = 2
            visited = set()
            for seed in seeds[:3]:
                queue: List[Tuple[str, List[str], float, int]] = [(seed, [seed], 1.0, 0)]
                while queue:
                    node, path, conf, depth = queue.pop(0)
                    if depth >= max_depth:
                        continue
                    for _, nbr, edge_data in graph.out_edges(node, data=True):
                        if nbr in visited:
                            continue
                        visited.add(nbr)
                        edge_conf = float(edge_data.get('confidence', 1.0))
                        new_conf = conf * edge_conf
                        new_path = path + [nbr]
                        if not path_matches_metapath(new_path):
                            continue
                        # Score combines confidence and depth decay
                        depth_score = 1.0 / (depth + 2)
                        node_data = graph.nodes[nbr]
                        score = new_conf * depth_score
                        results.append(SearchResult(
                            entity_id=nbr,
                            name=node_data.get('name', ''),
                            entity_type=node_data.get('label', ''),
                            description=node_data.get('description', ''),
                            score=score,
                            search_type=SearchType.GRAPH_TRAVERSAL,
                            reasoning=f"Traversal from seed with path len {len(new_path)} and confidence {new_conf:.2f}",
                            metadata={"path": new_path, "confidence": new_conf, "depth": depth + 1}
                        ))
                        queue.append((nbr, new_path, new_conf, depth + 1))

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"Graph traversal search failed: {e}")
            return []
    
    def logical_filter_search(self, query: str, context: QueryContext, limit: int = 10) -> List[SearchResult]:
        """Perform logical filtering based on structured criteria."""
        results = []
        
        try:
            # If no specific filters, try to infer from query
            if not context.filters:
                # Simple keyword-based filtering
                query_lower = query.lower()
                
                # Check for entity type keywords
                type_keywords = {
                    'scientist': 'Person',
                    'researcher': 'Person', 
                    'person': 'Person',
                    'people': 'Person',
                    'organization': 'Organization',
                    'company': 'Organization',
                    'place': 'Place',
                    'location': 'Place',
                    'city': 'Place',
                    'country': 'Place',
                    'event': 'Event',
                    'concept': 'Concept',
                    'movie': 'Film',
                    'film': 'Film',
                    'creative': 'CreativeWork',
                    'work': 'CreativeWork'
                }
                
                for keyword, entity_type in type_keywords.items():
                    if keyword in query_lower:
                        context.filters['label'] = entity_type
                        break
            
            # Filter nodes based on context filters
            for node_id, data in self.graph_builder.graph.nodes(data=True):
                matches_filters = True
                
                # Apply filters
                for filter_key, filter_value in context.filters.items():
                    if filter_key in data:
                        if isinstance(filter_value, str):
                            if filter_value.lower() not in str(data[filter_key]).lower():
                                matches_filters = False
                                break
                        elif data[filter_key] != filter_value:
                            matches_filters = False
                            break
                    else:
                        matches_filters = False
                        break
                
                if matches_filters:
                    # Calculate relevance score based on query similarity
                    name_similarity = 0.0
                    if 'name' in data:
                        name_lower = data['name'].lower()
                        query_lower = query.lower()
                        # Simple keyword matching
                        query_words = query_lower.split()
                        name_words = name_lower.split()
                        common_words = set(query_words) & set(name_words)
                        if query_words:
                            name_similarity = len(common_words) / len(query_words)
                    
                    results.append(SearchResult(
                        entity_id=node_id,
                        name=data.get('name', ''),
                        entity_type=data.get('label', ''),
                        description=data.get('description', ''),
                        score=0.5 + name_similarity * 0.5,  # Base score + similarity
                        search_type=SearchType.LOGICAL_FILTER,
                        reasoning=f"Matches logical filters: {context.filters}",
                        metadata={'filters_applied': context.filters}
                    ))
            
            # Sort by score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Logical filter search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, context: QueryContext, limit: int = 10) -> List[SearchResult]:
        """Perform hybrid search: dense + BM25 + traversal with RRF fusion and rerank."""
        # Retrieve separate lists
        dense = self.vector_similarity_search(query, context, max(10, limit))
        lexical = self.bm25_search(query, max(10, limit))
        traversal = self.graph_traversal_search(query, context, max(10, limit))

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion([dense, lexical, traversal], k=60, limit=max(20, limit))

        # Optional logical filter refinement
        if context.filters:
            filtered = self.logical_filter_search(query, context, max(20, limit))
            fused = self._reciprocal_rank_fusion([fused, filtered], k=60, limit=max(20, limit))

        # Reranking (cross-encoder if available)
        reranked = self._cross_encoder_rerank(query, fused, limit)
        return reranked
    
    def execute_query(self, query: str, limit: int = 10) -> AgentResponse:
        """Execute a query using dynamic agent selection."""
        import time
        start_time = time.time()
        
        # Analyze query to determine strategy
        context = self.analyze_query(query)
        
        # Execute search based on determined strategy
        if context.search_type == SearchType.VECTOR_SIMILARITY:
            results = self.vector_similarity_search(query, context, limit)
            strategy = "Vector Similarity Search"
        elif context.search_type == SearchType.GRAPH_TRAVERSAL:
            results = self.graph_traversal_search(query, context, limit)
            strategy = "Graph Traversal Search"
        elif context.search_type == SearchType.LOGICAL_FILTER:
            results = self.logical_filter_search(query, context, limit)
            strategy = "Logical Filter Search"
        else:  # HYBRID
            results = self.hybrid_search(query, context, limit)
            strategy = "Hybrid Search"
        
        # Calculate confidence based on result quality
        confidence = min(1.0, len(results) / limit) if results else 0.0
        if results:
            confidence *= max(result.score for result in results)
        
        execution_time = time.time() - start_time
        
        return AgentResponse(
            query=query,
            results=results,
            reasoning_chain=context.reasoning_steps + [f"Executed {strategy}", f"Found {len(results)} results"],
            search_strategy=strategy,
            confidence=confidence,
            execution_time=execution_time
        )
    
    def multi_step_reasoning(self, query: str, max_steps: int = 3) -> AgentResponse:
        """Perform multi-step reasoning for complex queries."""
        reasoning_steps = []
        all_results = []
        
        # Initial query execution
        initial_response = self.execute_query(query)
        reasoning_steps.extend(initial_response.reasoning_chain)
        all_results.extend(initial_response.results)
        
        # Iterative refinement based on initial results
        for step in range(1, max_steps):
            if not all_results:
                break
            
            # Analyze results to determine if refinement is needed
            refinement_prompt = f"""
            Based on the query "{query}" and these initial results:
            {[{'name': r.name, 'type': r.entity_type, 'description': r.description[:100]} for r in all_results[:3]]}
            
            Should we refine the search? If yes, suggest a refined query or different approach.
            Return JSON: {{"refine": true/false, "refined_query": "new query", "reasoning": "explanation"}}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": refinement_prompt}],
                    max_tokens=200,
                    temperature=0.2
                )
                
                refinement = json.loads(response.choices[0].message.content.strip())
                
                if refinement.get('refine', False):
                    refined_query = refinement.get('refined_query', query)
                    reasoning_steps.append(f"Step {step}: {refinement.get('reasoning', '')}")
                    
                    # Execute refined query
                    refined_response = self.execute_query(refined_query)
                    reasoning_steps.extend(refined_response.reasoning_chain)
                    all_results.extend(refined_response.results)
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Multi-step reasoning failed at step {step}: {e}")
                break
        
        # Deduplicate and rerank final results
        unique_results = {}
        for result in all_results:
            key = result.entity_id
            if key not in unique_results or result.score > unique_results[key].score:
                unique_results[key] = result
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return AgentResponse(
            query=query,
            results=final_results[:10],
            reasoning_chain=reasoning_steps,
            search_strategy="Multi-Step Reasoning",
            confidence=min(1.0, len(final_results) / 10),
            execution_time=sum([r.execution_time for r in [initial_response]] + [0.1] * (len(reasoning_steps) - 1))
        )


if __name__ == "__main__":
    # Test agentic retrieval system
    from .graph_builder import KnowledgeGraphBuilder
    
    # This would require actual setup with Neo4j/Qdrant
    # graph_builder = KnowledgeGraphBuilder()
    # retrieval_system = AgenticRetrievalSystem(graph_builder, "your-api-key")
    
    # Test query
    # response = retrieval_system.execute_query("Find physicists who worked at universities")
    # print(f"Query: {response.query}")
    # print(f"Strategy: {response.search_strategy}")
    # print(f"Results: {len(response.results)}")
    # print(f"Confidence: {response.confidence:.2f}")
    print("Agentic retrieval system module loaded successfully")
