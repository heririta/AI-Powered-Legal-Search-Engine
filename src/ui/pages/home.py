"""
Home page for the AI-Powered Legal Search Engine.

This module renders the main application interface with document upload,
search, and results display functionality.
"""

import streamlit as st
import time
from typing import List, Optional
import logging

from services.database import get_database_service
from services.vector_store import get_vector_store_service
from services.embedding import get_embedding_service
from ui.components.upload import render_upload_component
from ui.components.search import render_search_component
from ui.components.results import render_search_results
from ui.components.admin import render_admin_component

logger = logging.getLogger(__name__)


def apply_neutral_theme():
    """Apply custom CSS for neutral color theme with modern design."""
    st.markdown("""
    <style>
    /* Neutral Color Theme with Modern Design */

    /* Main Theme Colors */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .subtitle {
        color: #546e7a;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Feature Cards */
    .welcome-section {
        margin: 2rem 0;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }

    .feature-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e1e8ed;
        transition: all 0.3s ease;
        text-align: center;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        border-color: #90a4ae;
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .feature-card h3 {
        color: #2c3e50;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }

    .feature-card p {
        color: #546e7a;
        margin: 0;
        line-height: 1.6;
        font-size: 0.95rem;
    }

    /* Streamlit Component Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8fafc;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        color: #546e7a;
        font-weight: 500;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #2c3e50;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* Headers */
    .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
        font-weight: 600;
    }

    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Buttons */
    .stButton > button {
        background-color: #607d8b;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #546e7a;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    .stButton > button[kind="primary"] {
        background-color: #455a64;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #37474f;
    }

    /* Success/Info/Error styling */
    .stSuccess {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }

    .stInfo {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1565c0;
    }

    .stWarning {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
        color: #ef6c00;
    }

    .stError {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #c62828;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
        border-right: 1px solid #e1e8ed;
    }

    /* Tab Headers */
    .tab-header {
        background: linear-gradient(135deg, #f8fafc 0%, #e1e8ed 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .tab-header h2 {
        color: #2c3e50;
        margin: 0 0 0.5rem 0;
        font-size: 1.8rem;
        font-weight: 600;
    }

    .tab-header p {
        color: #546e7a;
        margin: 0.25rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }

    .section-header {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #607d8b;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .section-header h3 {
        color: #2c3e50;
        margin: 0;
        font-size: 1.4rem;
        font-weight: 600;
    }

    /* Document card styling */
    .document-card {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }

    .document-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-color: #90a4ae;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 6px;
        padding: 0.5rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #607d8b;
        box-shadow: 0 0 0 2px rgba(96, 125, 139, 0.2);
    }

    /* File uploader */
    .stFileUploader > div {
        background-color: #f8fafc;
        border: 2px dashed #e1e8ed;
        border-radius: 8px;
        padding: 2rem;
        transition: all 0.2s ease;
    }

    .stFileUploader > div:hover {
        border-color: #90a4ae;
        background-color: #f1f5f9;
    }

    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #607d8b;
    }

    /* Slider */
    .stSlider > div > div > div {
        background-color: #607d8b;
    }

    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #fff8e1 0%, #fff3e0 100%);
        border: 1px solid #ffcc02;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .info-card p {
        color: #ef6c00;
        margin: 0;
        font-weight: 500;
    }

    /* Search results styling */
    .search-result-card {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }

    .search-result-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-color: #90a4ae;
    }

    /* Highlight styling */
    mark {
        background-color: #fff3cd;
        color: #856404;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 500;
    }

    /* Remove emoji default styling */
    .element-container img {
        display: inline !important;
    }

    </style>
    """, unsafe_allow_html=True)


def render_home_page():
    """Render the main home page."""
    # Apply custom CSS for neutral color theme
    apply_neutral_theme()

    # Main header with improved styling
    st.markdown("""
    <div class="main-header">
        <h1>AI-Powered Legal Search Engine</h1>
        <p class="subtitle">Advanced semantic search for Indonesian legal documents</p>
    </div>
    """, unsafe_allow_html=True)

    # Welcome section with better styling
    st.markdown("""
    <div class="welcome-section">
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <h3>Upload Documents</h3>
                <p>Import Indonesian legal documents in PDF or TXT format for intelligent processing</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>Smart Search</h3>
                <p>Search using natural language queries powered by advanced semantic understanding</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3>AI-Powered Answers</h3>
                <p>Get intelligent responses with accurate legal citations and references</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create styled tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload", "🔍 Search", "📊 Statistics", "🔧 Admin"])

    with tab1:
        render_upload_tab()

    with tab2:
        render_search_tab()

    with tab3:
        render_statistics_tab()

    with tab4:
        render_admin_tab()


def render_upload_tab():
    """Render the document upload tab."""
    st.markdown("""
    <div class="tab-header">
        <h2>📄 Upload Legal Documents</h2>
        <p>Upload Indonesian legal documents (UU, PP, Permen, etc.) to make them searchable.</p>
        <p><strong>Supported formats:</strong> PDF, TXT (Maximum file size: 100MB)</p>
    </div>
    """, unsafe_allow_html=True)

    # Render upload component
    render_upload_component()

    # Display uploaded documents with improved styling
    st.markdown("""
    <div class="section-header">
        <h3>📋 Uploaded Documents</h3>
    </div>
    """, unsafe_allow_html=True)
    display_uploaded_documents()


def render_search_tab():
    """Render the search tab."""
    st.markdown("""
    <div class="tab-header">
        <h2>🔍 Semantic Search</h2>
        <p>Search through your uploaded legal documents using natural language queries.</p>
        <p>The system uses semantic search to find relevant legal clauses based on meaning.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if documents are available
    db_service = get_database_service()
    stats = db_service.get_database_stats()

    if stats.get('documents_count', 0) == 0:
        st.markdown("""
        <div class="info-card">
            <p>⚠️ No documents uploaded yet. Please upload documents in the Upload tab first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Render search component
    render_search_component()


def render_statistics_tab():
    """Render the statistics tab."""
    st.header("📊 Database Statistics")

    try:
        db_service = get_database_service()
        vector_service = get_vector_store_service()

        # Database statistics
        st.subheader("📄 Document Statistics")
        db_stats = db_service.get_database_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", db_stats.get('documents_count', 0))
        with col2:
            st.metric("Total Chunks", db_stats.get('chunks_count', 0))
        with col3:
            st.metric("Database Size", f"{db_stats.get('database_size_mb', 0):.2f} MB")

        # Vector index statistics
        st.subheader("🔍 Vector Index Statistics")
        index_stats = vector_service.get_index_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Indexed Vectors", index_stats.total_vectors)
        with col2:
            st.metric("Index Dimension", index_stats.dimension)
        with col3:
            st.metric("Index Size", f"{index_stats.index_size_mb:.2f} MB")

        # API status
        st.subheader("🔗 API Status")
        embedding_service = get_embedding_service()

        col1, col2 = st.columns(2)
        with col1:
            api_valid = embedding_service.validate_api_key()
            if api_valid:
                st.success("✅ Cohere API: Connected")
            else:
                st.error("❌ Cohere API: Failed")

        with col2:
            st.info(f"📊 Model: {embedding_service.model}")
            st.info(f"📏 Dimension: {embedding_service.dimension}")

        # Recent activity
        st.subheader("🕒 Recent Activity")
        display_recent_activity()

    except Exception as e:
        st.error(f"Failed to load statistics: {e}")


def display_uploaded_documents():
    """Display list of uploaded documents."""
    try:
        db_service = get_database_service()

        # Get all documents
        with db_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, filename, uu_title, file_size, file_type,
                       processing_status, chunk_count, upload_date, updated_at
                FROM legal_documents
                ORDER BY upload_date DESC
            """)
            documents = cursor.fetchall()

        if not documents:
            st.info("No documents uploaded yet.")
            return

        # Display documents
        for doc in documents:
            doc_data = dict(doc)

            # Document header
            status_emoji = {
                'pending': '⏳',
                'processing': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(doc_data['processing_status'], '❓')

            with st.expander(f"{status_emoji} {doc_data['filename']} ({doc_data['uu_title']})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**📁 Filename:** {doc_data['filename']}")
                    st.write(f"📏 **Size:** {doc_data['file_size'] / (1024*1024):.2f} MB")
                    st.write(f"📄 **Type:** {doc_data['file_type'].upper()}")

                with col2:
                    st.write(f"📊 **Status:** {doc_data['processing_status'].title()}")
                    st.write(f"🔤 **Chunks:** {doc_data['chunk_count']}")
                    st.write(f"📅 **Uploaded:** {doc_data['upload_date']}")

                # Error message if failed
                if doc_data['processing_status'] == 'failed':
                    st.error(f"❌ Error: {doc_data.get('processing_error', 'Unknown error')}")

                # Action buttons
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button(f"📄 View Details", key=f"details_{doc_data['id']}"):
                        display_document_details(doc_data['id'])

                with col2:
                    if st.button(f"🔍 Search in Document", key=f"search_{doc_data['id']}"):
                        st.session_state.search_filter_uu = doc_data['uu_title']
                        st.switch_page("app.py")

                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{doc_data['id']}"):
                        if st.session_state.get(f'confirm_delete_{doc_data["id"]}', False):
                            delete_document(doc_data['id'])
                            st.success("Document deleted successfully!")
                            st.rerun()
                        else:
                            st.session_state[f'confirm_delete_{doc_data["id"]}'] = True
                            st.warning("⚠️ Click again to confirm deletion")

    except Exception as e:
        st.error(f"Failed to load documents: {e}")


def display_document_details(document_id: int):
    """Display detailed information about a document."""
    try:
        db_service = get_database_service()

        # Get document details
        document = db_service.get_document_by_id(document_id)
        if not document:
            st.error("Document not found")
            return

        # Get chunks
        chunks = db_service.get_chunks_by_document(document_id)

        st.subheader(f"📄 Document Details: {document['uu_title']}")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Filename:** {document['filename']}")
            st.write(f"**File Size:** {document['file_size'] / (1024*1024):.2f} MB")
            st.write(f"**Upload Date:** {document['upload_date']}")

        with col2:
            st.write(f"**Status:** {document['processing_status'].title()}")
            st.write(f"**Total Chunks:** {len(chunks)}")
            st.write(f"**Last Updated:** {document['updated_at']}")

        # Display chunks
        if chunks:
            st.subheader("🔤 Legal Chunks")

            for i, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
                with st.expander(f"📖 {chunk['pasal']} {chunk['ayat']} - {chunk['uu_title']}"):
                    st.write(f"**Chapter:** {chunk['bab'] or 'N/A'}")
                    st.write(f"**Article:** {chunk['pasal']}")
                    st.write(f"**Clause:** {chunk['ayat']}")
                    if chunk['butir']:
                        st.write(f"**Sub-clause:** {chunk['butir']}")
                    st.write(f"**Text:** {chunk['ayat_text']}")

            if len(chunks) > 10:
                st.info(f"Showing 10 of {len(chunks)} chunks...")

    except Exception as e:
        st.error(f"Failed to display document details: {e}")


def delete_document(document_id: int):
    """Delete a document and its associated data."""
    try:
        db_service = get_database_service()
        vector_service = get_vector_store_service()

        with st.spinner("🗑️ Deleting document and updating search index..."):
            # Step 1: Get embedding IDs for this document
            embedding_ids_to_remove = db_service.get_document_chunk_embedding_ids(document_id)

            # Step 2: Remove vectors from vector store if any exist
            if embedding_ids_to_remove:
                logger.info(f"Removing {len(embedding_ids_to_remove)} vectors from search index...")
                vector_removal_success = vector_service.remove_vectors_by_document(embedding_ids_to_remove)
                if not vector_removal_success:
                    logger.warning("Failed to remove vectors from search index, but continuing with database deletion")

            # Step 3: Delete document from database (this will cascade delete chunks and search results)
            deletion_success = db_service.delete_document(document_id)

            if deletion_success:
                st.success(f"✅ Document deleted successfully!")
                st.info(f"📊 Removed {len(embedding_ids_to_remove)} entries from search index")

                # Clear any session state related to this document
                if 'selected_document' in st.session_state and st.session_state.selected_document == document_id:
                    st.session_state.selected_document = None
                if f'confirm_delete_{document_id}' in st.session_state:
                    del st.session_state[f'confirm_delete_{document_id}']
            else:
                st.error("❌ Failed to delete document")

    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        st.error(f"❌ Failed to delete document: {e}")


def display_recent_activity():
    """Display recent search activity."""
    try:
        db_service = get_database_service()

        with db_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT query_text, results_count, execution_time_ms, created_at
                FROM search_queries
                ORDER BY created_at DESC
                LIMIT 10
            """)
            queries = cursor.fetchall()

        if not queries:
            st.info("No recent search activity.")
            return

        for query in queries:
            q = dict(query)
            st.write(f"🔍 **{q['query_text'][:50]}...**")
            st.write(f"   Results: {q['results_count']} | Time: {q['execution_time_ms']}ms | {q['created_at']}")
            st.write("---")

    except Exception as e:
        st.error(f"Failed to load recent activity: {e}")


def render_admin_tab():
    """Render the admin tools tab."""
    render_admin_component()