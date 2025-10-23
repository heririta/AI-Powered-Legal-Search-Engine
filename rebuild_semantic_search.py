#!/usr/bin/env python3
"""
Final working script to rebuild semantic search functionality.
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import time
from services.database import get_database_service
from services.embedding import get_embedding_service
from services.vector_store import get_vector_store_service

def rebuild_semantic_search():
    """Rebuild semantic search by regenerating embeddings and rebuilding index."""
    print("=== REBUILDING SEMANTIC SEARCH FUNCTIONALITY ===")

    # 1. Initialize services
    print("\n1. INITIALIZING SERVICES...")
    db_service = get_database_service()
    embedding_service = get_embedding_service()
    vector_service = get_vector_store_service()

    # 2. Get chunks from database
    print("\n2. RETRIEVING CHUNKS FROM DATABASE...")
    with db_service.get_connection() as conn:
        cursor = conn.execute("""
            SELECT lc.id, lc.ayat_text, lc.uu_title, lc.bab, lc.pasal
            FROM legal_chunks lc
            WHERE lc.ayat_text IS NOT NULL
            AND LENGTH(lc.ayat_text) > 20
            ORDER BY lc.document_id, lc.chunk_order
        """)
        chunks = cursor.fetchall()

    print(f"   Found {len(chunks)} chunks with content")

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
    print("   Created new empty vector index")

    # 4. Process chunks and rebuild index
    print("\n4. PROCESSING CHUNKS AND BUILDING INDEX...")
    success_count = 0
    error_count = 0

    # Process in smaller batches to avoid rate limits
    batch_size = 3

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(chunks)-1)//batch_size + 1
        print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        for chunk in batch:
            try:
                # Prepare text
                text = chunk['ayat_text']
                if len(text) > 8000:
                    text = text[:8000] + "..."

                # Generate embedding with correct input type
                embedding = embedding_service.generate_single_embedding(text, "search_document")

                if embedding is not None:
                    # Convert to proper format
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)
                    else:
                        embedding = embedding.astype(np.float32)

                    # Ensure correct shape (should be 1D for FAISS, not 2D)
                    if embedding.ndim == 2 and embedding.shape[0] == 1:
                        embedding = embedding.flatten()  # Convert (1, 1024) to (1024,)

                    # Generate embedding ID
                    embedding_id = f"doc_{chunk['id']}_chunk_v2"

                    # Add to vector store
                    success = vector_service.add_single_vector(embedding, embedding_id)
                    if success:
                        # Update database
                        with db_service.get_connection() as conn:
                            conn.execute("""
                                UPDATE legal_chunks
                                SET embedding_id = ?
                                WHERE id = ?
                            """, (embedding_id, chunk['id']))
                            conn.commit()

                        success_count += 1
                        doc_info = f"{chunk['uu_title'][:30]}..."
                        if chunk['bab']:
                            doc_info += f" - {chunk['bab']}"
                        print(f"     + Chunk {chunk['id']}: {doc_info}")
                    else:
                        error_count += 1
                        print(f"     ✗ Failed to add chunk {chunk['id']} to index")
                else:
                    error_count += 1
                    print(f"     ✗ Failed to generate embedding for chunk {chunk['id']}")

            except Exception as e:
                error_count += 1
                print(f"     - Error processing chunk {chunk['id']}: {str(e)[:100]}")

        # Add delay between batches
        if i + batch_size < len(chunks):
            time.sleep(2)

    print(f"\n   Processing complete:")
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
    print("\n6. VERIFYING INDEX...")
    try:
        index_stats = vector_service.get_index_stats()
        print(f"   Index statistics:")
        print(f"   - Total vectors: {index_stats.total_vectors}")
        print(f"   - Dimension: {index_stats.dimension}")
        print(f"   - Size: {index_stats.index_size_mb:.2f} MB")

        if index_stats.total_vectors > 0:
            print(f"   SUCCESS: Index built with {index_stats.total_vectors} vectors!")
        else:
            print("   WARNING: Index is still empty")
            return False

    except Exception as e:
        print(f"   Error verifying index: {e}")

    # 7. Test semantic search
    print("\n7. TESTING SEMANTIC SEARCH...")
    try:
        test_queries = [
            "kewajiban wajib pajak",
            "persyaratan pendirian perusahaan",
            "hak dan kewajiban karyawan"
        ]

        for query in test_queries:
            print(f"   Testing query: '{query}'")
            query_embedding = embedding_service.generate_single_embedding(query, "search_query")

            if query_embedding is not None:
                search_results = vector_service.search(query_embedding, k=3)
                print(f"   Results: {len(search_results)} found")

                if search_results:
                    for j, result in enumerate(search_results, 1):
                        print(f"     {j}. Score: {result.score:.4f}")
                else:
                    print("   No results found")
            else:
                print("   Failed to generate query embedding")

        if search_results:
            print("\n   SUCCESS: Semantic search is working!")
        else:
            print("\n   WARNING: Search returned no results")

    except Exception as e:
        print(f"   Error testing search: {e}")

    # 8. Final summary
    print(f"\n=== SEMANTIC SEARCH REBUILD COMPLETE ===")
    print(f"Processed {len(chunks)} chunks")
    print(f"Successfully indexed: {success_count}")
    print(f"Vector index now has {index_stats.total_vectors} vectors")

    if success_count > 0:
        print("Semantic search functionality has been restored!")
        return True
    else:
        print("Failed to rebuild semantic search")
        return False

if __name__ == "__main__":
    rebuild_semantic_search()