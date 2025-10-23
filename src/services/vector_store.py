"""
Vector store service for the AI-Powered Legal Search Engine.

This module handles FAISS vector index operations including creation,
management, persistence, and search functionality.
"""

import faiss
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    id: str
    score: float
    vector: Optional[np.ndarray] = None


@dataclass
class IndexStats:
    """Statistics about the vector index."""
    total_vectors: int
    dimension: int
    index_size_mb: float
    is_trained: bool
    last_updated: Optional[str] = None


class VectorStoreService:
    """Service for FAISS vector operations."""

    def __init__(self):
        """Initialize vector store service."""
        self.config = get_config()
        self.index_path = self.config.faiss_index_path
        self.dimension = self.config.embedding_dimension
        self.index: Optional[faiss.Index] = None
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}

    def initialize_index(self) -> bool:
        """Initialize or load FAISS index."""
        try:
            # Ensure index directory exists
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

            if Path(self.index_path).exists():
                # Load existing index
                success = self._load_index()
                if success:
                    logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
                    return True
                else:
                    logger.warning("Failed to load existing index, creating new one")

            # Create new index
            self._create_new_index()
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            return False

    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Use IndexFlatL2 for exact search (suitable for our use case)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_to_index = {}
        self.index_to_id = {}

    def _load_index(self) -> bool:
        """Load existing FAISS index from disk."""
        try:
            # Load index
            self.index = faiss.read_index(self.index_path)

            # Load mapping files
            mapping_path = self.index_path.replace('.faiss', '_mapping.pkl')
            if Path(mapping_path).exists():
                with open(mapping_path, 'rb') as f:
                    mappings = pickle.load(f)
                    self.id_to_index = mappings.get('id_to_index', {})
                    self.index_to_id = mappings.get('index_to_id', {})

            # Validate dimension
            if self.index.d != self.dimension:
                logger.error(f"Index dimension mismatch: expected {self.dimension}, got {self.index.d}")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def save_index(self) -> bool:
        """Save FAISS index and mappings to disk."""
        try:
            if self.index is None:
                logger.warning("No index to save")
                return False

            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save mappings
            mapping_path = self.index_path.replace('.faiss', '_mapping.pkl')
            mappings = {
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id
            }
            with open(mapping_path, 'wb') as f:
                pickle.dump(mappings, f)

            logger.info(f"Index saved with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def add_vectors(self, vectors: List[np.ndarray], ids: List[str]) -> bool:
        """Add vectors to the index."""
        try:
            if self.index is None:
                logger.error("Index not initialized")
                return False

            if len(vectors) != len(ids):
                logger.error("Vectors and IDs must have same length")
                return False

            if not vectors:
                logger.warning("No vectors to add")
                return True

            # Validate vector dimensions
            for i, vector in enumerate(vectors):
                if vector.shape[0] != self.dimension:
                    logger.error(f"Vector {i} dimension mismatch: expected {self.dimension}, got {vector.shape[0]}")
                    return False

            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)

            # Add vectors to index
            start_idx = self.index.ntotal
            self.index.add(vectors_array)

            # Update mappings
            for i, vector_id in enumerate(ids):
                idx = start_idx + i
                self.id_to_index[vector_id] = idx
                self.index_to_id[idx] = vector_id

            logger.info(f"Added {len(vectors)} vectors to index")
            return True

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False

    def add_single_vector(self, vector: np.ndarray, vector_id: str) -> bool:
        """Add a single vector to the index."""
        return self.add_vectors([vector], [vector_id])

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        try:
            if self.index is None:
                logger.error("Index not initialized")
                return []

            if self.index.ntotal == 0:
                logger.warning("Index is empty")
                return []

            # Validate query vector
            if query_vector.shape[0] != self.dimension:
                logger.error(f"Query vector dimension mismatch: expected {self.dimension}, got {query_vector.shape[0]}")
                return []

            # Ensure k is not larger than index size
            k = min(k, self.index.ntotal)

            # Reshape query vector for FAISS
            query_vector = query_vector.reshape(1, -1).astype(np.float32)

            # Search
            distances, indices = self.index.search(query_vector, k)

            # Convert to results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                vector_id = self.index_to_id.get(idx)
                if vector_id:
                    # Convert L2 distance to similarity score (cosine-like)
                    similarity_score = 1 / (1 + distance)
                    result = VectorSearchResult(
                        id=vector_id,
                        score=similarity_score,
                        vector=None  # Don't return vectors to save memory
                    )
                    results.append(result)

            logger.info(f"Search completed, found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Failed to search index: {e}")
            return []

    def get_index_stats(self) -> IndexStats:
        """Get statistics about the index."""
        try:
            if self.index is None:
                return IndexStats(
                    total_vectors=0,
                    dimension=self.dimension,
                    index_size_mb=0.0,
                    is_trained=False
                )

            total_vectors = self.index.ntotal
            index_size_mb = 0.0

            if Path(self.index_path).exists():
                index_size_mb = Path(self.index_path).stat().st_size / (1024 * 1024)

            return IndexStats(
                total_vectors=total_vectors,
                dimension=self.dimension,
                index_size_mb=index_size_mb,
                is_trained=self.index.is_trained if hasattr(self.index, 'is_trained') else True
            )

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return IndexStats(
                total_vectors=0,
                dimension=self.dimension,
                index_size_mb=0.0,
                is_trained=False
            )

    def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the index (rebuild required)."""
        try:
            if vector_id not in self.id_to_index:
                logger.warning(f"Vector ID {vector_id} not found in index")
                return False

            logger.warning("FAISS IndexFlatL2 does not support individual removal. Index rebuild required.")
            # For IndexFlatL2, we would need to rebuild the entire index
            # This is expensive, so we'll implement this only if needed
            return False

        except Exception as e:
            logger.error(f"Failed to remove vector: {e}")
            return False

    def remove_vectors_by_document(self, embedding_ids_to_remove: List[str]) -> bool:
        """Remove vectors belonging to a specific document by rebuilding the index.

        Since FAISS IndexFlatL2 doesn't support individual removal,
        we rebuild the index excluding the specified vectors.
        """
        try:
            if not embedding_ids_to_remove:
                return True  # Nothing to remove

            logger.info(f"Removing {len(embedding_ids_to_remove)} vectors by rebuilding index...")

            # Get current vectors and IDs
            current_ids = list(self.id_to_index.keys())
            vectors_to_keep = []
            ids_to_keep = []

            # Collect vectors and IDs to keep (excluding ones to remove)
            for vector_id in current_ids:
                if vector_id not in embedding_ids_to_remove:
                    # Get vector from index
                    index_pos = self.id_to_index[vector_id]
                    vector = self.index.reconstruct(index_pos)
                    vectors_to_keep.append(vector)
                    ids_to_keep.append(vector_id)

            # Rebuild index with remaining vectors
            success = self.rebuild_index(vectors_to_keep, ids_to_keep)

            if success:
                logger.info(f"Successfully removed {len(embedding_ids_to_remove)} vectors. Index now has {len(vectors_to_keep)} vectors.")
            else:
                logger.error("Failed to rebuild index after vector removal")

            return success

        except Exception as e:
            logger.error(f"Failed to remove vectors by document: {e}")
            return False

    def rebuild_index(self, vectors: List[np.ndarray], ids: List[str]) -> bool:
        """Rebuild the entire index with new vectors."""
        try:
            logger.info("Rebuilding index...")

            # Create new index
            self._create_new_index()

            # Add all vectors
            success = self.add_vectors(vectors, ids)

            if success:
                # Save rebuilt index
                self.save_index()
                logger.info("Index rebuilt successfully")

            return success

        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            return False

    def validate_index(self) -> bool:
        """Validate index integrity."""
        try:
            if self.index is None:
                return False

            # Check if index has vectors
            if self.index.ntotal == 0:
                logger.warning("Index is empty")
                return True  # Empty index is valid

            # Check mapping consistency
            if len(self.id_to_index) != self.index.ntotal:
                logger.error("Mapping size mismatch with index size")
                return False

            # Check dimension consistency
            if self.index.d != self.dimension:
                logger.error(f"Index dimension mismatch: expected {self.dimension}, got {self.index.d}")
                return False

            logger.info("Index validation passed")
            return True

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False

    def backup_index(self, backup_path: str) -> bool:
        """Create a backup of the index."""
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy index file
            import shutil
            shutil.copy2(self.index_path, backup_path)

            # Copy mapping file
            mapping_path = self.index_path.replace('.faiss', '_mapping.pkl')
            backup_mapping_path = str(backup_path).replace('.faiss', '_mapping.pkl')
            if Path(mapping_path).exists():
                shutil.copy2(mapping_path, backup_mapping_path)

            logger.info(f"Index backed up to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup index: {e}")
            return False

    def restore_index(self, backup_path: str) -> bool:
        """Restore index from backup."""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Restore index file
            import shutil
            shutil.copy2(backup_path, self.index_path)

            # Restore mapping file
            mapping_path = self.index_path.replace('.faiss', '_mapping.pkl')
            backup_mapping_path = str(backup_path).replace('.faiss', '_mapping.pkl')
            if Path(backup_mapping_path).exists():
                shutil.copy2(backup_mapping_path, mapping_path)

            # Reload index
            success = self._load_index()
            if success:
                logger.info(f"Index restored from {backup_path}")
                return True
            else:
                logger.error("Failed to reload restored index")
                return False

        except Exception as e:
            logger.error(f"Failed to restore index: {e}")
            return False


# Global vector store service instance
vector_store_service: Optional[VectorStoreService] = None


def get_vector_store_service() -> VectorStoreService:
    """Get the global vector store service instance."""
    global vector_store_service
    if vector_store_service is None:
        vector_store_service = VectorStoreService()
    return vector_store_service