#!/usr/bin/env python3
"""
Debug semantic search functionality to identify why searches return no results.
"""

import sys
import os
sys.path.insert(0, 'src')

from services.database import get_database_service
from services.embedding import get_embedding_service
from services.vector_store import get_vector_store_service
from models.document import ProcessingStatus

def debug_semantic_search():
    """Debug the complete semantic search pipeline."""
    print("=== DEBUGGING SEMANTIC SEARCH PIPELINE ===")

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

        # Check chunks
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

        # Check embeddings
        cursor = conn.execute("""
            SELECT lc.id, lc.embedding_id, lc.content_text, lc.uu_title, lc.bab
            FROM legal_chunks lc
            WHERE lc.embedding_id IS NOT NULL
            LIMIT 5
        """)
        chunks_with_embeddings = cursor.fetchall()
        print(f"\n   Sample chunks with embeddings:")
        for chunk in chunks_with_embeddings:
            print(f"   - Chunk ID: {chunk['id']}, Embedding ID: {chunk['embedding_id'][:50]}...")
            print(f"     UU: {chunk['uu_title']}, BAB: {chunk['bab']}")
            print(f"     Content: {chunk['content_text'][:100]}...")

    # 2. Check vector store status
    print("\n2. VECTOR STORE STATUS:")
    try:
        vector_service = get_vector_store_service()
        index_info = vector_service.get_index_info()
        print(f"   Vector index loaded: {index_info}")

        # Test vector search with a known embedding
        if chunk_stats['chunks_with_embeddings'] > 0:
            # Get first embedding from database
            with db_service.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT lc.embedding_id
                    FROM legal_chunks lc
                    WHERE lc.embedding_id IS NOT NULL
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    test_embedding_id = result['embedding_id']
                    print(f"   Testing search with embedding: {test_embedding_id[:20]}...")

                    # Perform search
                    search_results = vector_service.search_by_id(test_embedding_id, k=3)
                    print(f"   Search results: {len(search_results)} found")
                    for i, result in enumerate(search_results, 1):
                        print(f"     {i}. ID: {result.id[:20]}..., Score: {result.score:.4f}")
                else:
                    print("   No embeddings found in database")
        else:
            print("   No chunks with embeddings in database")

    except Exception as e:
        print(f"   Vector store error: {e}")

    # 3. Test embedding generation
    print("\n3. EMBEDDING GENERATION TEST:")
    try:
        embedding_service = get_embedding_service()
        test_query = "kewajiban wajib pajak"
        print(f"   Testing query: '{test_query}'")

        test_embedding = embedding_service.generate_single_embedding(test_query, "debug_test")
        if test_embedding is not None:
            print(f"   ✅ Embedding generated successfully (shape: {test_embedding.shape})")

            # Test vector search with generated embedding
            if vector_service:
                print("   Testing vector search with generated embedding...")
                search_results = vector_service.search(test_embedding, k=5)
                print(f"   Vector search results: {len(search_results)} found")
                for i, result in enumerate(search_results, 1):
                    print(f"     {i}. ID: {result.id[:20]}..., Score: {result.score:.4f}")
        else:
            print("   ❌ Failed to generate embedding")

    except Exception as e:
        print(f"   Embedding generation error: {e}")

    # 4. Test complete search pipeline
    print("\n4. COMPLETE SEARCH PIPELINE TEST:")
    try:
        # Import search function
        from ui.components.search import perform_search

        # We can't directly test perform_search without streamlit context
        # But we can test the components
        print("   Testing individual components...")

        # Test chunk retrieval by embedding IDs
        if 'search_results' in locals() and search_results:
            embedding_ids = [result.id for result in search_results[:3]]
            print(f"   Testing chunk retrieval for IDs: {len(embedding_ids)} IDs")

            chunks_data = {}
            for embedding_id in embedding_ids:
                chunk = db_service.get_chunk_by_embedding_id(embedding_id)
                if chunk:
                    chunks_data[embedding_id] = chunk

            print(f"   Chunks retrieved: {len(chunks_data)}")
            for embedding_id, chunk in chunks_data.items():
                print(f"     - {chunk['uu_title']} - {chunk.get('bab', 'N/A')}")

    except Exception as e:
        print(f"   Search pipeline error: {e}")

    # 5. Check actual document content
    print("\n5. DOCUMENT CONTENT ANALYSIS:")
    with db_service.get_connection() as conn:
        cursor = conn.execute("""
            SELECT lc.content_text, lc.uu_title, lc.bab, lc.pasal
            FROM legal_chunks lc
            WHERE lc.content_text IS NOT NULL
            AND LENGTH(lc.content_text) > 50
            ORDER BY RANDOM()
            LIMIT 3
        """)
        sample_chunks = cursor.fetchall()

        print(f"   Sample document content:")
        for i, chunk in enumerate(sample_chunks, 1):
            print(f"     {i}. {chunk['uu_title']} - {chunk.get('bab', 'N/A')}")
            print(f"        Content: {chunk['content_text'][:200]}...")

            # Check if content contains tax-related keywords
            content_lower = chunk['content_text'].lower()
            if any(keyword in content_lower for keyword in ['pajak', 'wajib pajak', 'pembayaran pajak']):
                print(f"        ✅ Contains tax-related keywords")
            else:
                print(f"        ❌ No tax-related keywords found")

    # 6. Summary and recommendations
    print("\n6. SUMMARY AND RECOMMENDATIONS:")
    print(f"   - Documents in database: {len(docs)}")
    print(f"   - Total chunks: {chunk_stats['total_chunks']}")
    print(f"   - Chunks with embeddings: {chunk_stats['chunks_with_embeddings']}")

    if chunk_stats['chunks_with_embeddings'] == 0:
        print("   ❌ ISSUE: No chunks have embeddings - vector search cannot work")
        print("   💡 SOLUTION: Need to generate embeddings for existing chunks")
    elif chunk_stats['chunks_with_embeddings'] < chunk_stats['total_chunks']:
        print(f"   ⚠️  ISSUE: Only {chunk_stats['chunks_with_embeddings']}/{chunk_stats['total_chunks']} chunks have embeddings")
        print("   💡 SOLUTION: Generate embeddings for remaining chunks")
    else:
        print("   ✅ All chunks have embeddings")

    if 'search_results' in locals() and len(search_results) == 0:
        print("   ❌ ISSUE: Vector search returns no results")
        print("   💡 POSSIBLE CAUSES:")
        print("      - Low similarity threshold")
        print("      - Embedding dimension mismatch")
        print("      - Empty or corrupted vector index")
        print("      - Language mismatch between query and documents")

if __name__ == "__main__":
    debug_semantic_search()