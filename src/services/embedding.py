"""
Embedding service for the AI-Powered Legal Search Engine.

This module handles Cohere API integration for generating text embeddings.
"""

import cohere
import numpy as np
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRequest:
    """Request for generating embeddings."""
    texts: List[str]
    model: str = "embed-multilingual-v3.0"
    input_type: str = "search_document"


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""
    embeddings: List[List[float]]
    model: str
    tokens_used: int
    api_time_ms: int


class EmbeddingService:
    """Service for Cohere embedding generation."""

    def __init__(self):
        """Initialize embedding service."""
        self.config = get_config()
        self.api_key = self.config.cohere_api_key
        self.model = self.config.cohere_model
        self.batch_size = self.config.embedding_batch_size
        self.dimension = self.config.embedding_dimension

        # Initialize Cohere client
        self.client = cohere.Client(self.api_key)

    def validate_api_key(self) -> bool:
        """Validate Cohere API key."""
        try:
            # Test API with a simple embedding request
            response = self.client.embed(
                texts=["test"],
                model=self.model,
                input_type="search_document"
            )
            return True

        except Exception as e:
            logger.error(f"Cohere API key validation failed: {e}")
            return False

    def generate_embeddings(self, texts: List[str], input_type: str = "search_document") -> Optional[List[np.ndarray]]:
        """Generate embeddings for list of texts."""
        try:
            if not texts:
                return []

            # Validate input
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    logger.warning(f"Empty text at index {i}, skipping")
                    continue

            # Remove empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return []

            # Batch processing
            all_embeddings = []
            total_tokens = 0
            total_time = 0

            for i in range(0, len(valid_texts), self.batch_size):
                batch_texts = valid_texts[i:i + self.batch_size]
                batch_start = time.time()

                try:
                    # Generate embeddings for batch
                    response = self.client.embed(
                        texts=batch_texts,
                        model=self.model,
                        input_type=input_type
                    )

                    # Convert to numpy arrays
                    batch_embeddings = [
                        np.array(embedding, dtype=np.float32)
                        for embedding in response.embeddings
                    ]

                    all_embeddings.extend(batch_embeddings)

                    # Track usage
                    batch_time = (time.time() - batch_start) * 1000  # Convert to ms
                    total_time += batch_time

                    # Note: Cohere doesn't provide token count in response
                    # We'll estimate based on text length
                    batch_tokens = sum(len(text.split()) for text in batch_texts)
                    total_tokens += batch_tokens

                    logger.info(f"Generated embeddings for batch {i//self.batch_size + 1}: {len(batch_texts)} texts")

                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {i//self.batch_size + 1}: {e}")
                    # Add empty embeddings for failed batch to maintain order
                    batch_embeddings = [np.zeros(self.dimension, dtype=np.float32) for _ in batch_texts]
                    all_embeddings.extend(batch_embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings in {total_time:.2f}ms")
            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None

    def generate_single_embedding(self, text: str, input_type: str = "search_query") -> Optional[np.ndarray]:
        """Generate embedding for single text."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return None

            start_time = time.time()

            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type=input_type
            )

            embedding = np.array(response.embeddings[0], dtype=np.float32)
            api_time = (time.time() - start_time) * 1000

            logger.info(f"Generated single embedding in {api_time:.2f}ms")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate single embedding: {e}")
            return None

    def get_embedding_dimension(self) -> int:
        """Get embedding vector dimension."""
        return self.dimension

    def batch_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[str]]:
        """Split texts into batches for API calls."""
        if batch_size is None:
            batch_size = self.batch_size

        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append(batch)

        return batches

    def estimate_tokens(self, texts: List[str]) -> int:
        """Estimate token count for texts (rough approximation)."""
        total_tokens = 0
        for text in texts:
            # Rough estimation: 1 token ≈ 4 characters for English
            # For Indonesian text, this is a reasonable approximation
            total_tokens += len(text) // 4
        return total_tokens

    def validate_embedding_dimension(self, embedding: np.ndarray) -> bool:
        """Validate embedding dimension."""
        if embedding is None:
            return False

        if embedding.shape[0] != self.dimension:
            logger.error(f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}")
            return False

        return True

    def normalize_embeddings(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize embeddings to unit vectors."""
        normalized = []
        for embedding in embeddings:
            if embedding is None:
                continue

            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_embedding = embedding / norm
                normalized.append(normalized_embedding)
            else:
                logger.warning("Zero vector encountered, skipping normalization")
                normalized.append(embedding)

        return normalized

    def test_embedding_quality(self, test_texts: List[str]) -> Dict[str, Any]:
        """Test embedding generation quality with sample texts."""
        try:
            if not test_texts:
                test_texts = [
                    "This is a test sentence about legal documents.",
                    "Indonesian law requires proper documentation.",
                    "The constitution provides fundamental rights."
                ]

            start_time = time.time()

            # Generate embeddings
            embeddings = self.generate_embeddings(test_texts)
            if not embeddings:
                return {"success": False, "error": "Failed to generate test embeddings"}

            total_time = (time.time() - start_time) * 1000

            # Validate embeddings
            valid_embeddings = []
            for i, embedding in enumerate(embeddings):
                if self.validate_embedding_dimension(embedding):
                    valid_embeddings.append(embedding)
                else:
                    logger.error(f"Invalid embedding for test text {i}")

            # Calculate similarity matrix
            similarity_matrix = []
            for i in range(len(valid_embeddings)):
                row = []
                for j in range(len(valid_embeddings)):
                    if i <= j:
                        # Cosine similarity
                        similarity = np.dot(valid_embeddings[i], valid_embeddings[j])
                        row.append(float(similarity))
                    else:
                        # Use previously calculated value (symmetry)
                        row.append(similarity_matrix[j][i])
                similarity_matrix.append(row)

            return {
                "success": True,
                "total_texts": len(test_texts),
                "valid_embeddings": len(valid_embeddings),
                "total_time_ms": total_time,
                "avg_time_per_embedding_ms": total_time / len(valid_embeddings) if valid_embeddings else 0,
                "similarity_matrix": similarity_matrix,
                "dimension": self.dimension
            }

        except Exception as e:
            logger.error(f"Embedding quality test failed: {e}")
            return {"success": False, "error": str(e)}

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics (estimated)."""
        # Note: Cohere doesn't provide detailed usage stats like OpenAI
        # This is a placeholder for potential future implementation
        return {
            "model": self.model,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "api_key_valid": self.validate_api_key()
        }


# Global embedding service instance
embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service