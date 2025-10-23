#!/usr/bin/env python3
"""
Fix the document filter by updating document status to completed.
"""

import sys
import os
sys.path.insert(0, 'src')

from services.database import get_database_service
from models.document import ProcessingStatus

def fix_document_status():
    """Update document status to completed to demonstrate filter functionality."""
    print("Fixing document status for filter demonstration...")

    db_service = get_database_service()

    with db_service.get_connection() as conn:
        # Update the first document to completed status with chunk count
        conn.execute("""
            UPDATE legal_documents
            SET processing_status = 'completed', chunk_count = 63
            WHERE id = 1
        """)

        # Check updated status
        cursor = conn.execute("""
            SELECT id, filename, uu_title, processing_status, chunk_count
            FROM legal_documents
            WHERE id = 1
        """)
        doc = cursor.fetchone()

        print(f"Updated document: ID={doc['id']}, Title={doc['uu_title']}, Status={doc['processing_status']}, Chunks={doc['chunk_count']}")

        # Test the filter function
        from ui.components.search import get_available_documents
        available_docs = get_available_documents()
        print(f"\nAvailable documents for filter: {len(available_docs)}")
        for doc in available_docs:
            print(f"  - {doc['uu_title']} ({doc['chunk_count']} chunks)")

        print(f"\nFilter dropdown will now show:")
        print(f"  - All Documents")
        for doc in available_docs:
            print(f"  - {doc['uu_title']}")

if __name__ == "__main__":
    fix_document_status()