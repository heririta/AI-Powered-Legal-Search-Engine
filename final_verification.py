#!/usr/bin/env python3
"""
Verifikasi akhir bahwa filter sekarang menampilkan kedua dokumen.
"""

import sys
import os
sys.path.insert(0, 'src')

from services.database import get_database_service
from ui.components.search import get_available_documents

def final_verification():
    """Verifikasi akhir filter dokument."""
    print("=== VERIFIKASI AKHIR FILTER DOKUMEN ===")

    # 1. Cek status semua dokumen
    db_service = get_database_service()
    with db_service.get_connection() as conn:
        cursor = conn.execute("""
            SELECT id, filename, uu_title, processing_status, chunk_count
            FROM legal_documents
        """)
        all_docs = cursor.fetchall()

        print("\n1. STATUS SEMUA DOKUMEN:")
        for i, doc in enumerate(all_docs, 1):
            print(f"   {i}. {doc['filename']}")
            print(f"      UU Title: {doc['uu_title']}")
            print(f"      Status: {doc['processing_status']}")
            print(f"      Chunks: {doc['chunk_count']}")

    # 2. Test get_available_documents (fungsi yang digunakan oleh filter)
    print("\n2. HASIL FILTER (get_available_documents):")
    available_docs = get_available_documents()

    print(f"   Jumlah dokumen di filter: {len(available_docs)}")
    for i, doc in enumerate(available_docs, 1):
        print(f"   {i}. {doc['uu_title']} ({doc['chunk_count']} chunks)")

    # 3. Simulasi dropdown options
    print("\n3. OPSI DROPDOWN FILTER:")
    dropdown_options = ["All Documents"] + [doc['uu_title'] for doc in available_docs]
    for i, option in enumerate(dropdown_options, 1):
        print(f"   {i}. {option}")

    # 4. Kesimpulan
    print("\n4. KESIMPULAN:")
    print(f"   - Total dokumen di database: {len(all_docs)}")
    print(f"   - Dokumen di filter: {len(available_docs)}")
    print(f"   - Opsi dropdown: {len(dropdown_options)}")

    if len(all_docs) == len(available_docs):
        print("   ✅ SUKSES: Semua dokumen muncul di filter!")
        print("   ✅ Filter 'Filter by UU Title' sekarang menampilkan semua dokumen yang ada")
    else:
        print("   ❌ MASALAH: Tidak semua dokumen muncul di filter")

if __name__ == "__main__":
    final_verification()