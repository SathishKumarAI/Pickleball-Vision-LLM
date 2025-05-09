"""Vector store for pickleball analysis."""
import numpy as np
from typing import List, Dict, Any, Optional, Union
from ..core.config.config import Config

class VectorStore:
    """Stores and retrieves embeddings for pickleball analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize vector store.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self._setup_store()
        
    def _setup_store(self):
        """Setup vector store based on configuration."""
        try:
            if self.config.VECTOR_STORE_TYPE == "faiss":
                import faiss
                
                # Create index
                self.index = faiss.IndexFlatL2(self.config.EMBEDDING_DIM)
                
                # Load existing index if available
                index_path = self.config.VECTOR_STORE_DIR / "faiss.index"
                if index_path.exists():
                    self.index = faiss.read_index(str(index_path))
                    
            elif self.config.VECTOR_STORE_TYPE == "weaviate":
                import weaviate
                
                # Connect to Weaviate
                self.client = weaviate.Client(
                    url=f"http://{self.config.VECTOR_STORE_HOST}:{self.config.VECTOR_STORE_PORT}"
                )
                
                # Create schema if needed
                self._create_schema()
                
            else:
                raise ValueError(f"Unsupported vector store type: {self.config.VECTOR_STORE_TYPE}")
                
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            raise
            
    def _create_schema(self):
        """Create Weaviate schema."""
        try:
            # Define schema
            schema = {
                "classes": [{
                    "class": "PickleballFrame",
                    "vectorizer": "none",
                    "properties": [
                        {
                            "name": "frame_id",
                            "dataType": ["string"]
                        },
                        {
                            "name": "timestamp",
                            "dataType": ["date"]
                        },
                        {
                            "name": "detections",
                            "dataType": ["text"]
                        }
                    ]
                }]
            }
            
            # Create schema
            self.client.schema.create(schema)
            
        except Exception as e:
            print(f"Error creating schema: {e}")
            raise
            
    def add_embedding(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add embedding to store.
        
        Args:
            embedding: Embedding vector
            metadata: Associated metadata
        """
        try:
            if self.config.VECTOR_STORE_TYPE == "faiss":
                # Add to FAISS index
                self.index.add(embedding.reshape(1, -1))
                
                # Save index
                index_path = self.config.VECTOR_STORE_DIR / "faiss.index"
                faiss.write_index(self.index, str(index_path))
                
            elif self.config.VECTOR_STORE_TYPE == "weaviate":
                # Add to Weaviate
                self.client.data_object.create(
                    class_name="PickleballFrame",
                    data_object=metadata,
                    vector=embedding.tolist()
                )
                
        except Exception as e:
            print(f"Error adding embedding: {e}")
            raise
            
    def search(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings.
        
        Args:
            query: Query embedding
            k: Number of results to return
            
        Returns:
            List of results with metadata
        """
        try:
            if self.config.VECTOR_STORE_TYPE == "faiss":
                # Search FAISS index
                distances, indices = self.index.search(query.reshape(1, -1), k)
                
                # Return results
                return [
                    {
                        'distance': float(distances[0][i]),
                        'index': int(indices[0][i])
                    }
                    for i in range(k)
                ]
                
            elif self.config.VECTOR_STORE_TYPE == "weaviate":
                # Search Weaviate
                result = self.client.query.get(
                    "PickleballFrame",
                    ["frame_id", "timestamp", "detections"]
                ).with_near_vector({
                    "vector": query.tolist()
                }).with_limit(k).do()
                
                # Return results
                return result['data']['Get']['PickleballFrame']
                
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return []
            
    def clear(self):
        """Clear all stored embeddings."""
        try:
            if self.config.VECTOR_STORE_TYPE == "faiss":
                # Reset FAISS index
                self.index = faiss.IndexFlatL2(self.config.EMBEDDING_DIM)
                
                # Save empty index
                index_path = self.config.VECTOR_STORE_DIR / "faiss.index"
                faiss.write_index(self.index, str(index_path))
                
            elif self.config.VECTOR_STORE_TYPE == "weaviate":
                # Delete all objects
                self.client.schema.delete_all()
                
                # Recreate schema
                self._create_schema()
                
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            raise 