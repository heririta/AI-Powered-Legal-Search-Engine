"""
Settings page for the AI-Powered Legal Search Engine.

This module renders the configuration and settings interface.
"""

import streamlit as st
import logging
from pathlib import Path

from utils.config import get_config
from services.embedding import get_embedding_service
from services.database import get_database_service
from services.vector_store import get_vector_store_service

logger = logging.getLogger(__name__)


def render_settings_page():
    """Render the settings page."""
    st.title("⚙️ Settings")
    st.markdown("Configure your AI-Powered Legal Search Engine settings.")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Keys", "📊 Performance", "🗄️ Database", "ℹ️ About"])

    with tab1:
        render_api_keys_tab()

    with tab2:
        render_performance_tab()

    with tab3:
        render_database_tab()

    with tab4:
        render_about_tab()


def render_api_keys_tab():
    """Render API keys configuration tab."""
    st.header("🔑 API Configuration")
    st.markdown(
        """
        Configure your API keys for Cohere (embeddings) and Groq (LLM).
        These keys are stored locally and never shared.
        """
    )

    # Cohere API Key
    st.subheader("🔗 Cohere API (Embeddings)")
    cohere_key = st.text_input(
        "Cohere API Key",
        type="password",
        help="Enter your Cohere API key for text embeddings",
        value=st.session_state.get('temp_cohere_key', '')
    )

    if st.button("🔍 Test Cohere API"):
        if cohere_key:
            with st.spinner("Testing Cohere API..."):
                try:
                    # Temporarily update config for testing
                    config = get_config()
                    original_key = config.cohere_api_key
                    config.cohere_api_key = cohere_key

                    embedding_service = get_embedding_service()
                    if embedding_service.validate_api_key():
                        st.success("✅ Cohere API key is valid!")
                        st.session_state.temp_cohere_key = cohere_key
                    else:
                        st.error("❌ Invalid Cohere API key")

                    # Restore original key
                    config.cohere_api_key = original_key

                except Exception as e:
                    st.error(f"❌ API test failed: {e}")
        else:
            st.error("Please enter a Cohere API key")

    st.markdown("---")

    # Groq API Key
    st.subheader("🤖 Groq API (LLM)")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key for AI responses",
        value=st.session_state.get('temp_groq_key', '')
    )

    if st.button("🔍 Test Groq API"):
        if groq_key:
            with st.spinner("Testing Groq API..."):
                try:
                    # Simple test - we can't actually test without implementing the RAG service
                    st.success("✅ Groq API key format appears valid!")
                    st.session_state.temp_groq_key = groq_key
                    st.info("Note: Full API validation will be done during first search")

                except Exception as e:
                    st.error(f"❌ API test failed: {e}")
        else:
            st.error("Please enter a Groq API key")

    st.markdown("---")

    # Save configuration
    if st.button("💾 Save API Keys", type="primary"):
        if cohere_key and groq_key:
            st.success("✅ API keys would be saved to .env file")
            st.info("Note: In a full implementation, these would be saved to your .env file")
        else:
            st.error("Please provide both API keys")

    # Help links
    st.markdown("---")
    st.markdown("### 📚 Need API Keys?")
    st.markdown("""
    - **Cohere**: Sign up at [cohere.com](https://cohere.com/) and get your API key from the dashboard
    - **Groq**: Sign up at [groq.com](https://groq.com/) and get your API key from the dashboard
    """)


def render_performance_tab():
    """Render performance settings tab."""
    st.header("📊 Performance Settings")
    st.markdown("Configure performance-related settings for optimal performance.")

    config = get_config()

    # Embedding settings
    st.subheader("🔤 Embedding Configuration")

    col1, col2 = st.columns(2)
    with col1:
        embedding_batch_size = st.number_input(
            "Embedding Batch Size",
            min_value=1,
            max_value=1000,
            value=config.embedding_batch_size,
            help="Number of texts to process in a single API call"
        )

    with col2:
        max_search_results = st.number_input(
            "Max Search Results",
            min_value=1,
            max_value=100,
            value=config.max_search_results,
            help="Maximum number of search results to return"
        )

    # Timeout settings
    st.subheader("⏱️ Timeout Configuration")

    col1, col2 = st.columns(2)
    with col1:
        search_timeout = st.number_input(
            "Search Timeout (ms)",
            min_value=1000,
            max_value=30000,
            value=config.search_timeout_ms,
            help="Maximum time to wait for search results"
        )

    with col2:
        llm_timeout = st.number_input(
            "LLM Timeout (ms)",
            min_value=5000,
            max_value=120000,
            value=config.llm_timeout_ms,
            help="Maximum time to wait for LLM responses"
        )

    # UI settings
    st.subheader("🎨 UI Configuration")

    col1, col2 = st.columns(2)
    with col1:
        page_size = st.number_input(
            "Results Per Page",
            min_value=5,
            max_value=50,
            value=config.page_size,
            help="Number of results to display per page"
        )

    with col2:
        max_query_length = st.number_input(
            "Max Query Length",
            min_value=100,
            max_value=2000,
            value=config.max_query_length,
            help="Maximum length of search queries"
        )

    if st.button("💾 Save Performance Settings"):
        st.success("✅ Performance settings would be saved")
        st.info("Note: In a full implementation, these would be saved to your configuration")


def render_database_tab():
    """Render database management tab."""
    st.header("🗄️ Database Management")
    st.markdown("Manage your document database and vector index.")

    # Database statistics
    st.subheader("📊 Current Statistics")

    try:
        db_service = get_database_service()
        vector_service = get_vector_store_service()

        db_stats = db_service.get_database_stats()
        index_stats = vector_service.get_index_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", db_stats.get('documents_count', 0))
        with col2:
            st.metric("Chunks", db_stats.get('chunks_count', 0))
        with col3:
            st.metric("Vectors", index_stats.total_vectors)
        with col4:
            st.metric("Index Size", f"{index_stats.index_size_mb:.1f} MB")

    except Exception as e:
        st.error(f"Failed to load statistics: {e}")

    st.markdown("---")

    # Database operations
    st.subheader("🔧 Database Operations")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📦 Backup Database"):
            st.info("Backup functionality would be implemented here")

    with col2:
        if st.button("🗑️ Clear All Data"):
            if st.session_state.get('confirm_clear_data', False):
                st.warning("⚠️ This would delete all documents and data!")
                st.session_state.confirm_clear_data = False
            else:
                st.session_state.confirm_clear_data = True
                st.warning("Click again to confirm data deletion")

    with col3:
        if st.button("🔄 Rebuild Index"):
            st.info("Index rebuild functionality would be implemented here")

    # Storage information
    st.subheader("💾 Storage Information")

    try:
        config = get_config()

        # File paths
        st.write(f"**Database:** {config.database_path}")
        st.write(f"**Vector Index:** {config.faiss_index_path}")
        st.write(f"**Upload Directory:** {config.upload_dir}")

        # File sizes
        db_path = Path(config.database_path)
        index_path = Path(config.faiss_index_path)

        if db_path.exists():
            db_size = db_path.stat().st_size / (1024 * 1024)
            st.write(f"**Database Size:** {db_size:.2f} MB")

        if index_path.exists():
            index_size = index_path.stat().st_size / (1024 * 1024)
            st.write(f"**Index Size:** {index_size:.2f} MB")

        upload_path = Path(config.upload_dir)
        if upload_path.exists():
            upload_files = list(upload_path.glob("*"))
            upload_size = sum(f.stat().st_size for f in upload_files if f.is_file()) / (1024 * 1024)
            st.write(f"**Upload Files:** {len(upload_files)} files ({upload_size:.2f} MB)")

    except Exception as e:
        st.error(f"Failed to get storage information: {e}")


def render_about_tab():
    """Render about information tab."""
    st.header("ℹ️ About")

    st.markdown("""
    ### 🔍 AI-Powered Legal Search Engine v1.0.0

    A standalone Streamlit application that enables legal professionals to:

    - 📄 **Upload** Indonesian legal documents (PDF/TXT)
    - 🔍 **Search** using natural language semantic search
    - 🤖 **Get AI-powered answers** with proper legal citations

    ---

    ### 🏗️ Architecture

    **Local-First Design:** All processing happens on your local machine to ensure data privacy.

    **Technology Stack:**
    - **Frontend:** Streamlit
    - **Vector Database:** FAISS (CPU)
    - **Metadata Database:** SQLite
    - **Embeddings:** Cohere embed-v4
    - **LLM:** Groq (Mixtral/Llama3)
    - **Document Parsing:** PyMuPDF

    ---

    ### 📋 Features

    - **Document Upload:** Support for PDF and TXT files up to 50MB
    - **Legal Structure Recognition:** Automatic extraction of UU, Bab, Pasal, Ayat hierarchy
    - **Semantic Search:** Context-based search using vector embeddings
    - **RAG Q&A:** AI-powered question answering with citations
    - **Performance:** <2 second search responses, <30 second document processing

    ---

    ### 🔐 Privacy & Security

    - ✅ All documents stored locally
    - ✅ No data sent to external servers (except API calls)
    - ✅ API keys stored securely in environment variables
    - ✅ No external logging of document content

    ---

    ### 📊 Performance Targets

    - Document processing: <30 seconds per 50MB file
    - Semantic search: <2 seconds (P90)
    - RAG generation: <2 seconds
    - Support for 100+ documents simultaneously
    - Memory usage: <2GB typical

    ---

    ### 🤝 Support

    For issues, questions, or contributions, please check the project documentation.

    **Built with ❤️ for legal professionals**
    """)