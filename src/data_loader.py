"""
Data loader for DBpedia entities dataset.
Handles loading and preprocessing of the Qdrant/dbpedia-entities dataset.
"""

from datasets import load_dataset
from typing import Dict, List, Any, Iterator
import logging

logger = logging.getLogger(__name__)


class DBpediaDataLoader:
    """Loads and preprocesses the DBpedia entities dataset."""
    
    def __init__(self, dataset_name: str = "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K"):
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_dataset(self) -> None:
        """Load the DBpedia entities dataset."""
        logger.info(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        logger.info(f"Dataset loaded successfully. Size: {len(self.dataset['train'])} entities")
    
    def get_sample_batch(self, size: int = 10) -> List[Dict[str, Any]]:
        """Get a sample batch of entities for analysis."""
        if self.dataset is None:
            self.load_dataset()
        
        return self.dataset['train'].select(range(size))
    
    def get_entity_by_id(self, entity_id: str) -> Dict[str, Any]:
        """Get a specific entity by its ID."""
        if self.dataset is None:
            self.load_dataset()
        
        for entity in self.dataset['train']:
            if entity.get('id') == entity_id:
                return entity
        
        raise ValueError(f"Entity with ID {entity_id} not found")
    
    def iterate_entities(self, batch_size: int = 100) -> Iterator[List[Dict[str, Any]]]:
        """Iterate through entities in batches."""
        if self.dataset is None:
            self.load_dataset()
        
        total_size = len(self.dataset['train'])
        for i in range(0, total_size, batch_size):
            end_idx = min(i + batch_size, total_size)
            yield self.dataset['train'].select(range(i, end_idx))
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset structure."""
        if self.dataset is None:
            self.load_dataset()
        
        sample = self.dataset['train'][0]
        return {
            'total_entities': len(self.dataset['train']),
            'features': list(sample.keys()),
            'sample_entity': sample
        }


if __name__ == "__main__":
    # Quick test of the data loader
    loader = DBpediaDataLoader()
    info = loader.get_dataset_info()
    print("Dataset Info:")
    print(f"Total entities: {info['total_entities']}")
    print(f"Features: {info['features']}")
    print(f"Sample entity keys: {list(info['sample_entity'].keys())}")
