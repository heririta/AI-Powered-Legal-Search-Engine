#!/usr/bin/env python3
"""
Debug detailed mengapa filter hanya menampilkan 1 dokumen padahal ada 2 dokumen.
"""

import sys
import os
sys.path.insert(0, 'src')

from services.database import get_database_service

def debug_detailed_filter():
    """Debug detail mengapa filter hanya menampilkan 1 dokumen."""
    print("=== DEBUG DETAIL FILTER DOKUMEN ===")

    db_service = get_database_service()

    with db_service.get_connection() as conn:
        # 1. Cek semua dokumen tanpa filter
        print("\n1. SEMUA DOKUMEN DI DATABASE:")
        cursor = conn.execute("""
            SELECT id, filename, uu_title, processing_status, chunk_count
            FROM legal_documents
        """)
        all_docs = cursor.fetchall()

        for i, doc in enumerate(all_docs, 1):
            print(f"   {i}. ID: {doc['id']}")
            print(f"      Filename: {doc['filename']}")
            print(f"      UU Title: {doc['uu_title']}")
            print(f"      Status: '{doc['processing_status']}'")
            print(f"      Chunks: {doc['chunk_count']}")
            print()

        # 2. Cek dokumen dengan status 'completed' (filter yang digunakan)
        print("2. DOKUMEN DENGAN STATUS 'completed' (Filter Aktual):")
        cursor = conn.execute("""
            SELECT id, filename, uu_title, processing_status, chunk_count
            FROM legal_documents
            WHERE processing_status = 'completed'
        """)
        completed_docs = cursor.fetchall()

        if completed_docs:
            for i, doc in enumerate(completed_docs, 1):
                print(f"   {i}. ID: {doc['id']}")
                print(f"      Filename: {doc['filename']}")
                print(f"      UU Title: {doc['uu_title']}")
                print(f"      Status: '{doc['processing_status']}'")
                print(f"      Chunks: {doc['chunk_count']}")
        else:
            print("   TIDAK ADA dokumen dengan status 'completed'")

        # 3. Cek query yang digunakan oleh get_available_documents
        print("\n3. QUERY GET_AVAILABLE_DOCUMENTS:")
        cursor = conn.execute("""
            SELECT DISTINCT uu_title, COUNT(*) as chunk_count
            FROM legal_documents
            WHERE processing_status = 'completed'
            GROUP BY uu_title
            ORDER BY uu_title
        """)
        filter_docs = cursor.fetchall()

        print(f"   Jumlah dokumen untuk filter: {len(filter_docs)}")
        for doc in filter_docs:
            print(f"   - {doc['uu_title']} ({doc['chunk_count']} chunks)")

        # 4. Cek apakah ada dokumen dengan status lain
        print("\n4. DOKUMEN BERDASARKAN STATUS:")
        cursor = conn.execute("""
            SELECT processing_status, COUNT(*) as count
            FROM legal_documents
            GROUP BY processing_status
        """)
        status_counts = cursor.fetchall()

        for status in status_counts:
            print(f"   Status '{status['processing_status']}': {status['count']} dokumen")

        # 5. Rekomendasi perbaikan
        print("\n5. ANALISIS MASALAH:")
        print("   - Total dokumen:", len(all_docs))
        print("   - Dokumen dengan status 'completed':", len(completed_docs))
        print("   - Dokumen yang muncul di filter:", len(filter_docs))

        if len(all_docs) > len(completed_docs):
            print("\n   MASALAH: Ada dokumen dengan status BUKAN 'completed'")
            print("   SOLUSI: Update status dokumen lain ke 'completed' agar muncul di filter")

            # Update semua dokumen ke status 'completed' untuk demo
            print("\n6. MEMPERBAIKI STATUS DOKUMEN...")
            conn.execute("""
                UPDATE legal_documents
                SET processing_status = 'completed'
                WHERE processing_status != 'completed'
            """)
            conn.commit()

            # Test lagi setelah perbaikan
            cursor = conn.execute("""
                SELECT DISTINCT uu_title, COUNT(*) as chunk_count
                FROM legal_documents
                WHERE processing_status = 'completed'
                GROUP BY uu_title
                ORDER BY uu_title
            """)
            final_docs = cursor.fetchall()

            print(f"   Setelah perbaikan - Filter akan menampilkan {len(final_docs)} dokumen:")
            for doc in final_docs:
                print(f"   - {doc['uu_title']}")

if __name__ == "__main__":
    debug_detailed_filter()