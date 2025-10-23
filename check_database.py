#!/usr/bin/env python3
"""
Check database state and demonstrate document filter functionality.
"""

import sys
import os
sys.path.insert(0, 'src')

from services.database import get_database_service

def check_database():
    """Check current database state."""
    print("Checking database state...")

    db_service = get_database_service()

    with db_service.get_connection() as conn:
        # Check documents
        cursor = conn.execute("SELECT COUNT(*) as count FROM legal_documents")
        doc_count = cursor.fetchone()['count']
        print(f"Documents in database: {doc_count}")

        # Check chunks
        cursor = conn.execute("SELECT COUNT(*) as count FROM legal_chunks")
        chunk_count = cursor.fetchone()['count']
        print(f"Chunks in database: {chunk_count}")

        if doc_count > 0:
            # Show document details
            cursor = conn.execute("""
                SELECT id, filename, uu_title, processing_status, chunk_count
                FROM legal_documents
            """)
            docs = cursor.fetchall()
            print("\nDocuments:")
            for doc in docs:
                print(f"  - ID: {doc['id']}, File: {doc['filename']}, Title: {doc['uu_title']}, Status: {doc['processing_status']}, Chunks: {doc['chunk_count']}")

            # Test the filter function
            from ui.components.search import get_available_documents
            available_docs = get_available_documents()
            print(f"\nAvailable documents for filter: {len(available_docs)}")
            for doc in available_docs:
                print(f"  - {doc['uu_title']} ({doc['chunk_count']} chunks)")
        else:
            print("\nNo documents in database - filter will only show 'All Documents'")

if __name__ == "__main__":
    check_database()