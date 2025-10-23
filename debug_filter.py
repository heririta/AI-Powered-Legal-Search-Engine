#!/usr/bin/env python3
"""
Debug the filter issue by checking exact database values.
"""

import sys
import os
sys.path.insert(0, 'src')

from services.database import get_database_service

def debug_filter():
    """Debug the filter functionality."""
    print("Debugging filter functionality...")

    db_service = get_database_service()

    with db_service.get_connection() as conn:
        # Check all documents with their exact status values
        cursor = conn.execute("""
            SELECT id, filename, uu_title, processing_status, chunk_count
            FROM legal_documents
        """)
        docs = cursor.fetchall()

        print("All documents in database:")
        for doc in docs:
            print(f"  ID: {doc['id']}, Title: {doc['uu_title']}, Status: '{doc['processing_status']}' (type: {type(doc['processing_status'])}), Chunks: {doc['chunk_count']}")

        # Test the exact query used by get_available_documents
        print("\nTesting filter query:")
        cursor = conn.execute("""
            SELECT DISTINCT uu_title, COUNT(*) as chunk_count
            FROM legal_documents
            WHERE processing_status = 'completed'
            GROUP BY uu_title
            ORDER BY uu_title
        """)
        filtered_docs = cursor.fetchall()

        print(f"Documents with status 'completed': {len(filtered_docs)}")
        for doc in filtered_docs:
            print(f"  - {doc['uu_title']} ({doc['chunk_count']} chunks)")

        # Try with different status values
        print("\nTrying with status 'processing':")
        cursor = conn.execute("""
            SELECT DISTINCT uu_title, COUNT(*) as chunk_count
            FROM legal_documents
            WHERE processing_status = 'processing'
            GROUP BY uu_title
            ORDER BY uu_title
        """)
        processing_docs = cursor.fetchall()

        print(f"Documents with status 'processing': {len(processing_docs)}")
        for doc in processing_docs:
            print(f"  - {doc['uu_title']} ({doc['chunk_count']} chunks)")

        # Update status to make sure it works
        print("\nUpdating document 1 status to 'completed'...")
        conn.execute("""
            UPDATE legal_documents
            SET processing_status = 'completed'
            WHERE id = 1
        """)
        conn.commit()

        # Test again
        cursor = conn.execute("""
            SELECT DISTINCT uu_title, COUNT(*) as chunk_count
            FROM legal_documents
            WHERE processing_status = 'completed'
            GROUP BY uu_title
            ORDER BY uu_title
        """)
        final_docs = cursor.fetchall()

        print(f"After update - Documents with status 'completed': {len(final_docs)}")
        for doc in final_docs:
            print(f"  - {doc['uu_title']} ({doc['chunk_count']} chunks)")

if __name__ == "__main__":
    debug_filter()