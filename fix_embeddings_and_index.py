#!/usr/bin/env python3
"""
Fix embeddings and rebuild vector index by regenerating all embeddings.
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
from services.database import get_database_service
from services.embedding import get_embedding_service
from services.vector_store import get_vector_store_service

def fix_embeddings_and_index():
    """Regenerate embeddings for all chunks and rebuild vector index."""
    print("=== FIXING EMBEDDINGS AND REBUILDING INDEX ===")

    # 1. Initialize services
    print("\n1. INITIALIZING SERVICES...")
    db_service = get_database_service()
    embedding_service = get_embedding_service()
    vector_service = get_vector_store_service()

    # 2. Get all chunks that need embeddings
    print("\n2. RETRIEVING CHUNKS FROM DATABASE...")
    with db_service.get_connection() as conn:
        cursor = conn.execute("""
            SELECT lc.id, lc.ayat_text, lc.uu_title, lc.embedding_id
            FROM legal_chunks lc
            WHERE lc.ayat_text IS NOT NULL
            AND LENGTH(lc.ayat_text) > 10
            ORDER BY lc.document_id, lc.chunk_order
        """)
        chunks = cursor.fetchall()

    print(f"   Found {len(chunks)} chunks to process")

    if len(chunks) == 0:
        print("   No chunks found to process")
        return False

    # 3. Initialize vector store
    print("\n3. INITIALIZING VECTOR STORE...")
    if not vector_service.initialize_index():
        print("   Failed to initialize vector store")
        return False

    # Create new empty index
    vector_service._create_new_index()
    print("   Created new empty index")

    # 4. Regenerate embeddings for all chunks
    print("\n4. REGENERATING EMBEDDINGS...")
    success_count = 0
    error_count = 0

    # Process in smaller batches to avoid rate limits
    batch_size = 5

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")

        for chunk in batch:
            try:
                # Prepare text for embedding
                text = chunk['ayat_text']
                if len(text) > 8000:  # Limit text length
                    text = text[:8000] + "..."

                # Generate new embedding
                embedding = embedding_service.generate_single_embedding(text, "document_chunk")

                if embedding is not None:
                    # Convert embedding to numpy array
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)
                    else:
                        embedding = embedding.astype(np.float32)

                    # Reshape if needed
                    if embedding.ndim == 1:
                        embedding = embedding.reshape(1, -1)

                    # Generate embedding ID
                    embedding_id = f"doc_{chunk['id']}_chunk_fixed"

                    # Add to vector store
                    success = vector_service.add_embedding(embedding_id, embedding)
                    if success:
                        # Update database with new embedding info
                        with db_service.get_connection() as conn:
                            conn.execute("""
                                UPDATE legal_chunks
                                SET embedding_id = ?, embedding_vector = ?
                                WHERE id = ?
                            """, (embedding_id, embedding.tobytes(), chunk['id']))
                            conn.commit()

                        success_count += 1
                        print(f"     ✓ Processed chunk {chunk['id']}: {chunk['uu_title'][:30]}...")
                    else:
                        error_count += 1
                        print(f"     ✗ Failed to add embedding for chunk {chunk['id']}")
                else:
                    error_count += 1
                    print(f"     ✗ Failed to generate embedding for chunk {chunk['id']}")

            except Exception as e:
                error_count += 1
                print(f"     ✗ Error processing chunk {chunk['id']}: {e}")

        # Add small delay between batches to avoid rate limits
        if i + batch_size < len(chunks):
            import time
            time.sleep(1)

    print(f"\n   Embedding generation complete:")
    print(f"   - Successful: {success_count}")
    print(f"   - Errors: {error_count}")

    # 5. Save vector index
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

        if index_stats.total_vectors > 0:
            print(f"   SUCCESS: Vector index now has {index_stats.total_vectors} vectors!")
        else:
            print("   WARNING: Vector index is still empty")

    except Exception as e:
        print(f"   Error verifying index: {e}")

    # 7. Test search functionality
    if index_stats.total_vectors > 0:
        print("\n7. TESTING SEARCH FUNCTIONALITY...")
        try:
            test_query = "kewajiban wajib pajak"
            print(f"   Testing query: '{test_query}'")

            test_embedding = embedding_service.generate_single_embedding(test_query, "search_query")
            if test_embedding is not None:
                # Test search
                search_results = vector_service.search(test_embedding, k=5)
                print(f"   Search results: {len(search_results)} found")

                if search_results:
                    print("   Top 3 results:")
                    for i, result in enumerate(search_results[:3], 1):
                        print(f"     {i}. ID: {result.id[:30]}..., Score: {result.score:.4f}")
                    print("   SUCCESS: Semantic search is now working!")
                else:
                    print("   WARNING: Search returned no results")
            else:
                print("   Failed to generate test embedding")

        except Exception as e:
            print(f"   Error testing search: {e}")

    print(f"\n=== EMBEDDING FIX COMPLETE ===")
    print(f"Successfully processed {success_count} chunks")
    return success_count > 0

if __name__ == "__main__":
    fix_embeddings_and_index()