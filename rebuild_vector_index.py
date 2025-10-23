#!/usr/bin/env python3
"""
Rebuild the vector index with existing embeddings from the database.
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pickle
from services.database import get_database_service
from services.vector_store import get_vector_store_service

def rebuild_vector_index():
    """Rebuild vector index with existing embeddings from database."""
    print("=== REBUILDING VECTOR INDEX ===")

    # 1. Initialize services
    print("\n1. INITIALIZING SERVICES...")
    db_service = get_database_service()
    vector_service = get_vector_store_service()

    # 2. Initialize vector store
    print("\n2. INITIALIZING VECTOR STORE...")
    if not vector_service.initialize_index():
        print("   Failed to initialize vector store")
        return False

    # 3. Get all chunks with embeddings from database
    print("\n3. RETRIEVING EMBEDDINGS FROM DATABASE...")
    with db_service.get_connection() as conn:
        cursor = conn.execute("""
            SELECT lc.embedding_id, lc.embedding_vector
            FROM legal_chunks lc
            WHERE lc.embedding_id IS NOT NULL
            AND lc.embedding_vector IS NOT NULL
            ORDER BY lc.id
        """)
        chunks_with_embeddings = cursor.fetchall()

    print(f"   Found {len(chunks_with_embeddings)} chunks with embedding vectors")

    if len(chunks_with_embeddings) == 0:
        print("   No chunks with embedding vectors found")
        return False

    # 4. Clear existing index and rebuild
    print("\n4. REBUILDING VECTOR INDEX...")

    # Create new index
    vector_service._create_new_index()
    print("   Created new empty index")

    # Add embeddings to index
    success_count = 0
    batch_size = 10  # Process in batches to avoid memory issues

    for i in range(0, len(chunks_with_embeddings), batch_size):
        batch = chunks_with_embeddings[i:i+batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(chunks_with_embeddings)-1)//batch_size + 1} ({len(batch)} chunks)...")

        for chunk in batch:
            try:
                # Deserialize embedding vector
                embedding_vector = pickle.loads(chunk['embedding_vector'])
                embedding_id = chunk['embedding_id']

                # Convert to numpy array if needed
                if not isinstance(embedding_vector, np.ndarray):
                    embedding_vector = np.array(embedding_vector, dtype=np.float32)

                # Reshape if needed
                if embedding_vector.ndim == 1:
                    embedding_vector = embedding_vector.reshape(1, -1)

                # Add to vector store
                success = vector_service.add_embedding(embedding_id, embedding_vector)
                if success:
                    success_count += 1
                else:
                    print(f"     Failed to add embedding {embedding_id[:20]}...")

            except Exception as e:
                print(f"     Error processing embedding {chunk['embedding_id'][:20]}...: {e}")

    # 5. Save index
    print("\n5. SAVING VECTOR INDEX...")
    try:
        vector_service.save_index()
        print("   Vector index saved successfully")
    except Exception as e:
        print(f"   Failed to save vector index: {e}")
        return False

    # 6. Verify index
    print("\n6. VERIFYING REBUILT INDEX...")
    try:
        index_stats = vector_service.get_index_stats()
        print(f"   Index statistics after rebuild:")
        print(f"   - Total vectors: {index_stats.total_vectors}")
        print(f"   - Dimension: {index_stats.dimension}")
        print(f"   - Index size: {index_stats.index_size_mb:.2f} MB")
        print(f"   - Is trained: {index_stats.is_trained}")

        if index_stats.total_vectors == success_count:
            print(f"   SUCCESS: All {success_count} embeddings added to index!")
        else:
            print(f"   WARNING: Expected {success_count}, but index has {index_stats.total_vectors}")

    except Exception as e:
        print(f"   Error verifying index: {e}")
        return False

    # 7. Test search functionality
    print("\n7. TESTING SEARCH FUNCTIONALITY...")
    try:
        # Create a test query embedding
        from services.embedding import get_embedding_service
        embedding_service = get_embedding_service()

        test_query = "kewajiban wajib pajak"
        print(f"   Testing query: '{test_query}'")

        test_embedding = embedding_service.generate_single_embedding(test_query, "rebuild_test")
        if test_embedding is not None:
            # Test search
            search_results = vector_service.search(test_embedding, k=5)
            print(f"   Search results: {len(search_results)} found")

            if search_results:
                print("   Top 3 results:")
                for i, result in enumerate(search_results[:3], 1):
                    print(f"     {i}. ID: {result.id[:20]}..., Score: {result.score:.4f}")
                print("   SUCCESS: Vector search is now working!")
            else:
                print("   WARNING: Search returned no results")
        else:
            print("   Failed to generate test embedding")

    except Exception as e:
        print(f"   Error testing search: {e}")

    print(f"\n=== VECTOR INDEX REBUILD COMPLETE ===")
    print(f"Successfully added {success_count} embeddings to vector index")
    return True

if __name__ == "__main__":
    rebuild_vector_index()