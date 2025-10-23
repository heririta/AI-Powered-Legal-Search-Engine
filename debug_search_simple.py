#!/usr/bin/env python3
"""
Simple debug script to investigate semantic search issues.
"""

import sys
import os
sys.path.insert(0, 'src')

from services.database import get_database_service
from services.embedding import get_embedding_service
from services.vector_store import get_vector_store_service

def debug_search_simple():
    """Simple debug of semantic search functionality."""
    print("=== SIMPLE SEMANTIC SEARCH DEBUG ===")

    # 1. Check database status
    print("\n1. DATABASE STATUS:")
    db_service = get_database_service()

    with db_service.get_connection() as conn:
        # Check documents
        cursor = conn.execute("""
            SELECT id, filename, uu_title, processing_status, chunk_count
            FROM legal_documents
        """)
        docs = cursor.fetchall()
        print(f"   Total documents: {len(docs)}")
        for doc in docs:
            print(f"   - {doc['uu_title']} (ID: {doc['id']}, Status: {doc['processing_status']}, Chunks: {doc['chunk_count']})")

        # Check chunks count
        cursor = conn.execute("""
            SELECT COUNT(*) as total_chunks,
                   COUNT(DISTINCT document_id) as documents_with_chunks,
                   COUNT(DISTINCT embedding_id) as chunks_with_embeddings
            FROM legal_chunks
        """)
        chunk_stats = cursor.fetchone()
        print(f"\n   Chunk Statistics:")
        print(f"   - Total chunks: {chunk_stats['total_chunks']}")
        print(f"   - Documents with chunks: {chunk_stats['documents_with_chunks']}")
        print(f"   - Chunks with embeddings: {chunk_stats['chunks_with_embeddings']}")

    # 2. Check vector store
    print("\n2. VECTOR STORE STATUS:")
    try:
        vector_service = get_vector_store_service()
        if not vector_service.initialize_index():
            print("   ❌ Failed to initialize vector store")
            return

        index_stats = vector_service.get_index_stats()
        print(f"   Vector index stats:")
        print(f"   - Total vectors: {index_stats.total_vectors}")
        print(f"   - Dimension: {index_stats.dimension}")
        print(f"   - Index size: {index_stats.index_size_mb:.2f} MB")
        print(f"   - Is trained: {index_stats.is_trained}")

        if index_stats.total_vectors == 0:
            print("   ❌ ISSUE: Vector index is empty!")
            return

    except Exception as e:
        print(f"   Vector store error: {e}")
        return

    # 3. Test embedding generation
    print("\n3. EMBEDDING GENERATION TEST:")
    try:
        embedding_service = get_embedding_service()
        test_query = "kewajiban wajib pajak"
        print(f"   Testing query: '{test_query}'")

        test_embedding = embedding_service.generate_single_embedding(test_query, "debug_test")
        if test_embedding is not None:
            print(f"   ✅ Embedding generated successfully (shape: {test_embedding.shape})")
        else:
            print("   ❌ Failed to generate embedding")
            return
    except Exception as e:
        print(f"   Embedding generation error: {e}")
        return

    # 4. Test vector search
    print("\n4. VECTOR SEARCH TEST:")
    try:
        search_results = vector_service.search(test_embedding, k=10)
        print(f"   Search found {len(search_results)} results")

        if search_results:
            print("   Top 5 results:")
            for i, result in enumerate(search_results[:5], 1):
                print(f"     {i}. ID: {result.id[:20]}..., Score: {result.score:.4f}")
        else:
            print("   ❌ NO RESULTS FOUND - This is the main issue!")

    except Exception as e:
        print(f"   Vector search error: {e}")
        return

    # 5. Test chunk retrieval
    if search_results:
        print("\n5. CHUNK RETRIEVAL TEST:")
        try:
            embedding_ids = [result.id for result in search_results[:3]]
            print(f"   Testing retrieval for {len(embedding_ids)} embedding IDs")

            chunks_data = {}
            for embedding_id in embedding_ids:
                chunk = db_service.get_chunk_by_embedding_id(embedding_id)
                if chunk:
                    chunks_data[embedding_id] = chunk

            print(f"   Retrieved {len(chunks_data)} chunks")
            for embedding_id, chunk in chunks_data.items():
                print(f"     - {chunk['uu_title']} - {chunk.get('bab', 'N/A')}")
                print(f"       Content: {chunk['ayat_text'][:100]}...")

        except Exception as e:
            print(f"   Chunk retrieval error: {e}")

    # 6. Check sample document content
    print("\n6. SAMPLE DOCUMENT CONTENT:")
    try:
        with db_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT lc.ayat_text, lc.uu_title, lc.bab, lc.pasal
                FROM legal_chunks lc
                WHERE lc.ayat_text IS NOT NULL
                AND LENGTH(lc.ayat_text) > 50
                ORDER BY RANDOM()
                LIMIT 3
            """)
            sample_chunks = cursor.fetchall()

            print(f"   Sample chunks:")
            for i, chunk in enumerate(sample_chunks, 1):
                print(f"     {i}. {chunk['uu_title']} - {chunk.get('bab', 'N/A')}")
                content_preview = chunk['ayat_text'][:200] + "..." if len(chunk['ayat_text']) > 200 else chunk['ayat_text']
                print(f"        Content: {content_preview}")

                # Check for tax-related keywords
                content_lower = chunk['ayat_text'].lower()
                tax_keywords = ['pajak', 'wajib pajak', 'pembayaran pajak', 'perpajakan', 'npwp']
                found_keywords = [kw for kw in tax_keywords if kw in content_lower]
                if found_keywords:
                    print(f"        ✅ Contains tax keywords: {', '.join(found_keywords)}")
                else:
                    print(f"        ❌ No tax keywords found")

    except Exception as e:
        print(f"   Content analysis error: {e}")

    # 7. Diagnosis
    print("\n7. DIAGNOSIS:")
    if not search_results:
        print("   ❌ PRIMARY ISSUE: Vector search returns no results")
        print("   Possible causes:")
        print("   - Vector index is empty or corrupted")
        print("   - Embedding dimension mismatch")
        print("   - Index was built with different embeddings")
        print("   - Low similarity scores")

        # Check if we can rebuild the index
        print("\n   SUGGESTED FIXES:")
        print("   1. Rebuild the vector index with current embeddings")
        print("   2. Check embedding dimensions consistency")
        print("   3. Lower similarity threshold in search")
    else:
        print("   ✅ Vector search is working")

    print(f"\n   SUMMARY:")
    print(f"   - Documents: {len(docs)}")
    print(f"   - Chunks: {chunk_stats['total_chunks']}")
    print(f"   - Embeddings: {chunk_stats['chunks_with_embeddings']}")
    print(f"   - Vector search results: {len(search_results) if search_results else 0}")

if __name__ == "__main__":
    debug_search_simple()