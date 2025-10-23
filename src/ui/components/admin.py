"""
Admin tools component for the AI-Powered Legal Search Engine.

This module provides administrative functions for system maintenance.
"""

import streamlit as st
import logging
import time
from typing import Optional

from services.database import get_database_service
from services.embedding import get_embedding_service
from services.vector_store import get_vector_store_service

logger = logging.getLogger(__name__)


def render_admin_component():
    """Render the admin tools interface."""
    st.markdown("### 🔧 Admin Tools")

    # Check current status
    st.markdown("#### 📊 System Status")

    # Database status
    try:
        db_service = get_database_service()
        with db_service.get_connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) as count FROM legal_documents").fetchone()
            chunk_count = conn.execute("SELECT COUNT(*) as count FROM legal_chunks").fetchone()
            embedded_count = conn.execute("SELECT COUNT(*) as count FROM legal_chunks WHERE embedding_id IS NOT NULL").fetchone()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Documents", doc_count['count'])
            with col2:
                st.metric("📝 Chunks", chunk_count['count'])
            with col3:
                st.metric("🔐 Embedded", embedded_count['count'])
    except Exception as e:
        st.error(f"❌ Database error: {e}")

    # Vector store status
    try:
        vector_service = get_vector_store_service()
        if vector_service.initialize_index():
            stats = vector_service.get_index_stats()
            st.metric("🔍 Vectors in Index", stats.total_vectors)

            if stats.total_vectors == 0:
                st.warning("⚠️ Vector index is empty - Rebuild recommended")
            else:
                st.success("✅ Vector index is healthy")
        else:
            st.error("❌ Vector index initialization failed")
    except Exception as e:
        st.error(f"❌ Vector store error: {e}")

    st.markdown("---")

    # Admin actions
    st.markdown("#### 🛠️ Admin Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Rebuild Vector Index", type="primary", help="Rebuild the entire vector index from database chunks"):
            rebuild_vector_index()

    with col2:
        if st.button("📊 Show Statistics", help="Display detailed system statistics"):
            show_system_statistics()

    st.markdown("---")

    # Warning section
    st.markdown("#### ⚠️ Important Notes")
    st.warning("""
    **Rebuilding Vector Index:**
    - This will rebuild the entire FAISS index from all database chunks
    - Process may take 10-20 minutes depending on data size
    - System will be temporarily unavailable during rebuild
    - Only click this button if vector index is empty or corrupted
    """)


def rebuild_vector_index():
    """Rebuild the vector index from all database chunks."""
    try:
        # Show progress placeholder
        progress_bar = st.progress(0, "Initializing rebuild process...")
        status_text = st.empty()

        status_text.info("🔧 Starting vector index rebuild...")

        # Initialize services
        db_service = get_database_service()
        embedding_service = get_embedding_service()
        vector_service = get_vector_store_service()

        # Get all chunks with embeddings
        status_text.info("📋 Retrieving chunks from database...")
        progress_bar.progress(0.1, "Retrieving chunks...")

        with db_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT lc.id, lc.embedding_id, lc.ayat_text, lc.uu_title
                FROM legal_chunks lc
                WHERE lc.embedding_id IS NOT NULL
                ORDER BY lc.id
            """)
            chunks_data = cursor.fetchall()

        total_chunks = len(chunks_data)
        status_text.info(f"📊 Found {total_chunks} chunks to process")

        if total_chunks == 0:
            status_text.warning("⚠️ No chunks with embeddings found in database")
            return

        # Create new vector index
        status_text.info("🆕 Creating new vector index...")
        progress_bar.progress(0.2, "Creating vector index...")

        if not vector_service.initialize_index():
            status_text.error("❌ Failed to initialize vector index")
            return

        # Process chunks in batches
        batch_size = 3
        total_batches = (total_chunks + batch_size - 1) // batch_size

        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = chunks_data[i:batch_end]

            # Update progress
            progress = 0.2 + (0.7 * (batch_end / total_chunks))
            progress_bar.progress(progress, f"Processing batch {i//batch_size + 1}/{total_batches}...")
            status_text.info(f"🔄 Processing {len(batch_chunks)} chunks...")

            # Generate embeddings for batch
            chunk_texts = [chunk['ayat_text'] for chunk in batch_chunks]
            embeddings = embedding_service.generate_embeddings(chunk_texts)

            if embeddings:
                # Add to vector index
                embedding_ids = [chunk['embedding_id'] for chunk in batch_chunks]
                vector_service.add_vectors(embeddings, embedding_ids)
            else:
                logger.warning(f"Failed to generate embeddings for batch {i//batch_size + 1}")

        # Save vector index
        status_text.info("💾 Saving vector index...")
        progress_bar.progress(0.95, "Saving vector index...")

        vector_service.save_index()

        # Complete
        progress_bar.progress(1.0, "Rebuild complete!")
        status_text.success("✅ Vector index rebuilt successfully!")

        # Show final stats
        time.sleep(1)
        if vector_service.initialize_index():
            stats = vector_service.get_index_stats()
            st.success(f"🎉 Rebuild complete! Vector index now contains {stats.total_vectors} vectors")

        # Clear progress indicators
        time.sleep(2)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Vector index rebuild failed: {e}")
        logger.error(f"Vector index rebuild failed: {e}")


def show_system_statistics():
    """Show detailed system statistics."""
    try:
        st.markdown("#### 📈 Detailed System Statistics")

        # Database statistics
        st.markdown("**Database Statistics:**")
        db_service = get_database_service()

        with db_service.get_connection() as conn:
            # Documents by status
            cursor = conn.execute("""
                SELECT processing_status, COUNT(*) as count
                FROM legal_documents
                GROUP BY processing_status
            """)
            status_counts = cursor.fetchall()

            for status in status_counts:
                st.write(f"• {status['processing_status']}: {status['count']} documents")

            # Chunks statistics
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_chunks,
                    COUNT(CASE WHEN embedding_id IS NOT NULL THEN 1 END) as embedded_chunks,
                    COUNT(CASE WHEN embedding_id IS NULL THEN 1 END) as non_embedded_chunks
                FROM legal_chunks
            """)
            chunk_stats = cursor.fetchone()

            st.write(f"• Total chunks: {chunk_stats['total_chunks']}")
            st.write(f"• With embeddings: {chunk_stats['embedded_chunks']}")
            st.write(f"• Without embeddings: {chunk_stats['non_embedded_chunks']}")

            # Document types
            cursor = conn.execute("""
                DISTINCT uu_title, COUNT(*) as chunk_count
                FROM legal_documents ld
                JOIN legal_chunks lc ON ld.id = lc.document_id
                WHERE lc.embedding_id IS NOT NULL
                GROUP BY uu_title
                ORDER BY chunk_count DESC
            """)
            doc_types = cursor.fetchall()

            st.markdown("**Documents with Embeddings:**")
            for doc in doc_types:
                st.write(f"• {doc['uu_title']}: {doc['chunk_count']} chunks")

        # Vector store statistics
        st.markdown("**Vector Store Statistics:**")
        vector_service = get_vector_store_service()

        if vector_service.initialize_index():
            stats = vector_service.get_index_stats()
            st.write(f"• Total vectors: {stats.total_vectors}")
            st.write(f"• Index dimension: {stats.dimension}")
            st.write(f"• Index path: {stats.index_path}")

        # Performance metrics
        st.markdown("**Performance Metrics:**")
        st.write("• Embedding generation: Cohere API")
        st.write("• Vector search: FAISS (Facebook AI Similarity Search)")
        st.write("• Database: SQLite")

    except Exception as e:
        st.error(f"❌ Failed to show statistics: {e}")
        logger.error(f"Statistics display failed: {e}")