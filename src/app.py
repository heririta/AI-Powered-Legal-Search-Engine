"""
Main Streamlit application for the AI-Powered Legal Search Engine.

This is the entry point for the application that provides the web interface
for document upload, search, and AI-powered Q&A functionality.
"""

import streamlit as st
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config
from services.database import get_database_service
from services.vector_store import get_vector_store_service
from services.embedding import get_embedding_service
from ui.pages.home import render_home_page
from ui.pages.settings import render_settings_page


def setup_logging():
    """Setup application logging."""
    config = get_config()

    # Create logs directory if it doesn't exist
    Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def initialize_services():
    """Initialize all required services."""
    try:
        # Initialize database
        db_service = get_database_service()
        if not db_service.initialize_database():
            st.error("Failed to initialize database")
            return False

        # Initialize vector store
        vector_service = get_vector_store_service()
        if not vector_service.initialize_index():
            st.error("Failed to initialize vector store")
            return False

        # Validate embedding service
        embedding_service = get_embedding_service()
        if not embedding_service.validate_api_key():
            st.error("Failed to validate Cohere API key")
            return False

        return True

    except Exception as e:
        st.error(f"Service initialization failed: {e}")
        return False


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("🔍 Legal Search Engine")

    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Home", "⚙️ Settings"],
        index=0
    )

    # API Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status")

    try:
        config = get_config()
        db_service = get_database_service()
        vector_service = get_vector_store_service()

        # Database stats
        db_stats = db_service.get_database_stats()
        st.sidebar.metric("📄 Documents", db_stats.get('documents_count', 0))
        st.sidebar.metric("🔤 Chunks", db_stats.get('chunks_count', 0))

        # Vector index stats
        index_stats = vector_service.get_index_stats()
        st.sidebar.metric("📈 Vectors", index_stats.total_vectors)
        st.sidebar.metric("💾 Index Size", f"{index_stats.index_size_mb:.1f} MB")

    except Exception as e:
        st.sidebar.error(f"Status unavailable: {e}")

    return page


def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(
        page_title="AI-Powered Legal Search Engine",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Setup logging
    setup_logging()

    # Initialize services
    if not initialize_services():
        st.stop()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if page == "🏠 Home":
        render_home_page()
    elif page == "⚙️ Settings":
        render_settings_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        AI-Powered Legal Search Engine v1.0.0 | Built with Streamlit, FAISS, Cohere & Groq
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()