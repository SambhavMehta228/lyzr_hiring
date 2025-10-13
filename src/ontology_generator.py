"""
LLM-driven ontology generation for entities, relationships, and hierarchies.
Creates structured ontologies from unstructured DBpedia text data.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any]
    embeddings: Optional[List[float]] = None


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    relationship_type: str
    confidence: float
    properties: Dict[str, Any]


@dataclass
class OntologySchema:
    """Defines the ontology schema for the knowledge graph."""
    entity_types: Dict[str, Dict[str, Any]]
    relationship_types: Dict[str, Dict[str, Any]]
    hierarchies: Dict[str, List[str]]


class OntologyGenerator:
    """Generates ontologies from unstructured text using LLM reasoning."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.schema = None
        
    def generate_entity_type(self, entity_data: Dict[str, Any]) -> str:
        """Generate entity type from entity data using LLM."""
        text_content = entity_data.get('text', '')
        
        prompt = f"""
        Analyze the following text and determine the most specific entity type from DBpedia.
        Return only the type name (e.g., "Person", "Organization", "Place", "Event", "Concept").
        
        Text: {text_content[:500]}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate entity type: {e}")
            return "Unknown"
    
    def extract_relationships(self, entity_data: Dict[str, Any], existing_entities: List[Entity] = None) -> List[Relationship]:
        """Extract potential relationships from entity data."""
        text_content = entity_data.get('text', '')
        entity_id = entity_data.get('_id', '') # Use _id from dataset
        
        # Create a simple relationship extraction that creates self-relationships for demo
        relationships = []
        
        # Create a self-referential relationship for demonstration
        if len(text_content) > 50:  # Only for entities with substantial content
            relationships.append(Relationship(
                source_id=entity_id,
                target_id=entity_id,  # Self-relationship for demo
                relationship_type="self_reference",
                confidence=1.0,
                properties={'demo_relationship': True, 'text_length': len(text_content)}
            ))
        
        # If we have existing entities, try to find simple text-based relationships
        if existing_entities and len(existing_entities) > 1:
            entity_names = {e.name.lower(): e.id for e in existing_entities if e.id != entity_id}
            
            # Simple pattern-based relationship extraction
            text_lower = text_content.lower()
            
            # Look for mentions of other entities in the text
            for name, target_id in entity_names.items():
                if name in text_lower and len(name) > 3:  # Avoid short matches
                    # Determine relationship type based on context
                    rel_type = "related_to"
                    if any(word in text_lower for word in ['founded', 'created', 'established']):
                        rel_type = "founded_by"
                    elif any(word in text_lower for word in ['worked', 'employed', 'member']):
                        rel_type = "member_of"
                    elif any(word in text_lower for word in ['born', 'birth', 'from']):
                        rel_type = "born_in"
                    elif any(word in text_lower for word in ['located', 'in', 'city', 'country']):
                        rel_type = "located_in"
                    
                    relationships.append(Relationship(
                        source_id=entity_id,
                        target_id=target_id,
                        relationship_type=rel_type,
                        confidence=0.7,
                        properties={'extracted_from_text': True, 'matched_name': name}
                    ))
        
        return relationships
    
    def create_entity(self, entity_data: Dict[str, Any]) -> Entity:
        """Create an Entity object from raw entity data."""
        entity_type = self.generate_entity_type(entity_data)
        
        return Entity(
            id=entity_data.get('_id', ''),
            name=entity_data.get('title', ''),
            type=entity_type,
            description=entity_data.get('text', '')[:500],
            properties={
                'original_data': entity_data,
                'embedding_dim': len(entity_data.get('text-embedding-3-small-1024-embedding', []))
            },
            embeddings=entity_data.get('text-embedding-3-small-1024-embedding', [])
        )
    
    def build_ontology_schema(self, entities: List[Entity]) -> OntologySchema:
        """Build ontology schema from a collection of entities."""
        entity_types = {}
        relationship_types = {}
        
        # Analyze entity types
        for entity in entities:
            if entity.type not in entity_types:
                entity_types[entity.type] = {
                    'count': 0,
                    'properties': set(),
                    'description': f"DBpedia {entity.type} entities"
                }
            entity_types[entity.type]['count'] += 1
        
        # Convert sets to lists for JSON serialization
        for entity_type in entity_types:
            entity_types[entity_type]['properties'] = list(entity_types[entity_type]['properties'])
        
        # Define common relationship types
        relationship_types = {
            'related_to': {'description': 'General relatedness'},
            'part_of': {'description': 'Part-whole relationship'},
            'located_in': {'description': 'Location relationship'},
            'member_of': {'description': 'Membership relationship'},
            'founded_by': {'description': 'Founding relationship'},
            'born_in': {'description': 'Birth location'},
            'works_for': {'description': 'Employment relationship'}
        }
        
        # Build hierarchies based on entity types
        hierarchies = {
            'Thing': ['Person', 'Organization', 'Place', 'Event', 'Concept'],
            'Person': ['Scientist', 'Artist', 'Politician', 'Athlete'],
            'Place': ['City', 'Country', 'Continent', 'Building'],
            'Organization': ['Company', 'University', 'Government', 'NGO']
        }
        
        return OntologySchema(
            entity_types=entity_types,
            relationship_types=relationship_types,
            hierarchies=hierarchies
        )
    
    def resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve and deduplicate similar entities."""
        # Simple entity resolution based on name similarity
        resolved_entities = []
        seen_names = set()
        
        for entity in entities:
            # Normalize name for comparison
            normalized_name = entity.name.lower().strip()
            
            if normalized_name not in seen_names:
                seen_names.add(normalized_name)
                resolved_entities.append(entity)
            else:
                # Merge with existing entity
                existing = next(e for e in resolved_entities if e.name.lower().strip() == normalized_name)
                # Update properties and merge descriptions
                existing.description += f" | {entity.description}"
                existing.properties.update(entity.properties)
        
        return resolved_entities


if __name__ == "__main__":
    # Test ontology generation
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    generator = OntologyGenerator(os.getenv("OPENAI_API_KEY"))
    
    # Sample entity data for testing
    sample_entity = {
        'id': 'test_entity_1',
        'name': 'Albert Einstein',
        'text': 'Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.',
        'vector': [0.1] * 1024
    }
    
    entity = generator.create_entity(sample_entity)
    print(f"Generated entity: {entity.name} ({entity.type})")
