"""
Search interface component for the AI-Powered Legal Search Engine.

This module provides the search input interface with filtering options
and query validation.
"""

import streamlit as st
import logging
from typing import Optional, List, Dict, Any

from services.database import get_database_service
from services.embedding import get_embedding_service
from services.vector_store import get_vector_store_service

logger = logging.getLogger(__name__)


def render_search_component():
    """Render the search interface."""
    # Search input section
    st.subheader("🔍 Search Legal Documents")

    # Handle example query from session state
    if 'example_query' in st.session_state:
        query = st.text_input(
            "Enter your legal question or search query:",
            value=st.session_state.example_query,
            placeholder="Example: Apa saja persyaratan untuk mendirikan perusahaan?",
            help="Ask questions in natural language about Indonesian law, regulations, or legal procedures",
            key="search_query"
        )
        # Clear the example query after using it
        del st.session_state.example_query
    else:
        query = st.text_input(
            "Enter your legal question or search query:",
            placeholder="Example: Apa saja persyaratan untuk mendirikan perusahaan?",
            help="Ask questions in natural language about Indonesian law, regulations, or legal procedures",
            key="search_query"
        )

    # Get available documents for filters
    available_docs = get_available_documents()

    # Get current UU filter value
    current_uu_filter = st.session_state.get('uu_filter', 'All Documents')

    # Get available chapters based on selected UU filter
    available_babs = get_available_chapters(current_uu_filter)

    # Filters section
    with st.expander("🎛️ Search Filters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # UU Title filter
            uu_filter = st.selectbox(
                "Filter by UU Title:",
                options=["All Documents"] + [doc['uu_title'] for doc in available_docs],
                index=0,
                key="uu_filter"
            )

        with col2:
            # Chapter/Bab filter
            bab_options = ["All Chapters"] + available_babs
            bab_filter = st.selectbox(
                "Filter by Chapter (Bab):",
                options=bab_options,
                index=0,
                key="bab_filter"
            )

        # Advanced search options
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider(
                "Maximum Results:",
                min_value=1,
                max_value=20,
                value=5,
                key="max_results"
            )

        with col2:
            similarity_threshold = st.slider(
                "Similarity Threshold:",
                min_value=0.1,
                max_value=1.0,
                value=0.4,
                step=0.1,
                key="similarity_threshold"
            )

    # Search button
    search_button = st.button(
        "🔍 Search",
        type="primary",
        use_container_width=True,
        disabled=not query or not query.strip()
    )

    # Display search examples
    if not query:
        st.markdown("---")
        st.markdown("### 💡 Search Examples")
        example_queries = [
            "What are the requirements for establishing a foreign investment company?",
            "Apa saja kewajiban wajib pajak sesuai UU KUP?",
            "How to obtain a business license in Indonesia?",
            "Apa saja jenis perusahaan yang diakui di Indonesia?",
            "What are the labor rights for employees?"
        ]

        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(f"💭 {example[:50]}...", key=f"example_{i}"):
                    # Store the example query in session state before widget creation
                    st.session_state.example_query = example
                    st.rerun()

    # Perform search
    if search_button and query and query.strip():
        perform_search(query, uu_filter, bab_filter, max_results, similarity_threshold)


def get_available_documents() -> List[Dict[str, Any]]:
    """Get list of available documents for filtering."""
    try:
        db_service = get_database_service()

        with db_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT ld.uu_title, COUNT(lc.id) as chunk_count
                FROM legal_documents ld
                INNER JOIN legal_chunks lc ON ld.id = lc.document_id
                WHERE lc.embedding_id IS NOT NULL
                GROUP BY ld.uu_title
                ORDER BY ld.uu_title
            """)
            return [dict(row) for row in cursor.fetchall()]

    except Exception as e:
        logger.error(f"Failed to get available documents: {e}")
        return []


def get_available_chapters(uu_filter: str = None) -> List[str]:
    """Get list of available chapters from documents, filtered by UU title if specified."""
    try:
        db_service = get_database_service()

        with db_service.get_connection() as conn:
            if uu_filter and uu_filter != "All Documents":
                # Filter chapters by specific UU title (only embedded documents)
                cursor = conn.execute("""
                    SELECT DISTINCT lc.bab
                    FROM legal_chunks lc
                    JOIN legal_documents ld ON lc.document_id = ld.id
                    WHERE lc.bab IS NOT NULL
                    AND lc.bab != ''
                    AND ld.uu_title = ?
                    AND lc.embedding_id IS NOT NULL
                    ORDER BY lc.bab
                """, (uu_filter,))
            else:
                # Get all chapters from all documents (only embedded documents)
                cursor = conn.execute("""
                    SELECT DISTINCT lc.bab
                    FROM legal_chunks lc
                    WHERE lc.bab IS NOT NULL
                    AND lc.bab != ''
                    AND lc.embedding_id IS NOT NULL
                    ORDER BY lc.bab
                """)
            return [row['bab'] for row in cursor.fetchall()]

    except Exception as e:
        logger.error(f"Failed to get available chapters: {e}")
        return []


def perform_search(query: str, uu_filter: str, bab_filter: str, max_results: int, similarity_threshold: float):
    """Perform semantic search with the given parameters."""
    try:
        # Show searching indicator
        with st.spinner("🔍 Searching..."):
            start_time = time.time()

            # Validate query
            if len(query.strip()) > 500:
                st.error("❌ Query too long. Please limit to 500 characters.")
                return

            # Initialize services
            embedding_service = get_embedding_service()
            vector_service = get_vector_store_service()
            db_service = get_database_service()

            # Generate query embedding
            logger.info(f"Generating embedding for query: {query[:100]}...")
            query_embedding = embedding_service.generate_single_embedding(query, "search_query")

            if query_embedding is None:
                st.error("❌ Failed to process search query. Please try again.")
                return

            # Perform vector search
            logger.info("Performing vector search...")
            search_results = vector_service.search(query_embedding, k=max_results)

            if not search_results:
                st.warning("🔍 No results found. Try different keywords or check if documents are uploaded.")
                return

            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results
                if result.score >= similarity_threshold
            ]

            if not filtered_results:
                st.warning(f"🔍 No results found above similarity threshold {similarity_threshold}. Try lowering the threshold.")
                return

            # Get chunk details from database
            chunk_ids = [result.id for result in filtered_results]
            chunks_data = get_chunks_by_embedding_ids(chunk_ids)

            if not chunks_data:
                st.error("❌ Failed to retrieve search results. Please try again.")
                return

            # Combine results with chunk data
            combined_results = []
            for result in filtered_results:
                chunk_data = chunks_data.get(result.id)
                if chunk_data:
                    combined_results.append({
                        'chunk': chunk_data,
                        'similarity_score': result.score,
                        'rank': len(combined_results) + 1
                    })

            # Apply additional filters
            if uu_filter != "All Documents":
                combined_results = [
                    result for result in combined_results
                    if result['chunk']['uu_title'] == uu_filter
                ]

            if bab_filter != "All Chapters":
                combined_results = [
                    result for result in combined_results
                    if result['chunk']['bab'] == bab_filter
                ]

            if not combined_results:
                st.warning("🔍 No results found matching all filters.")
                return

            # Calculate search time
            search_time = (time.time() - start_time) * 1000

            # Save search to database
            save_search_to_database(query, combined_results, search_time)

            # Display results
            from .results import render_search_results
            render_search_results(query, combined_results, search_time)

    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        logger.error(f"Search failed: {e}")


def get_chunks_by_embedding_ids(embedding_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get chunk data by embedding IDs."""
    try:
        db_service = get_database_service()
        chunks_data = {}

        for embedding_id in embedding_ids:
            chunk = db_service.get_chunk_by_embedding_id(embedding_id)
            if chunk:
                chunks_data[embedding_id] = chunk

        return chunks_data

    except Exception as e:
        logger.error(f"Failed to get chunks by embedding IDs: {e}")
        return {}


def save_search_to_database(query: str, results: List[Dict], search_time_ms: int):
    """Save search query and results to database."""
    try:
        db_service = get_database_service()

        # Save search query
        query_data = {
            'query_text': query,
            'filter_uu_title': st.session_state.get('uu_filter'),
            'filter_bab': st.session_state.get('bab_filter'),
            'search_type': 'semantic',
            'results_count': len(results),
            'execution_time_ms': int(search_time_ms)
        }

        search_query_id = db_service.save_search_query(query_data)
        if not search_query_id:
            return

        # Save search results
        results_data = []
        for result in results:
            results_data.append({
                'search_query_id': search_query_id,
                'chunk_id': result['chunk']['id'],
                'similarity_score': result['similarity_score'],
                'rank': result['rank']
            })

        db_service.save_search_results(results_data)
        logger.info(f"Saved search query {search_query_id} with {len(results)} results")

    except Exception as e:
        logger.error(f"Failed to save search to database: {e}")


def display_search_history():
    """Display recent search history."""
    try:
        db_service = get_database_service()

        with db_service.get_connection() as conn:
            cursor = conn.execute("""
                SELECT query_text, results_count, execution_time_ms, created_at
                FROM search_queries
                ORDER BY created_at DESC
                LIMIT 5
            """)
            recent_searches = cursor.fetchall()

        if recent_searches:
            st.markdown("### 🔍 Recent Searches")
            for search in recent_searches:
                search_data = dict(search)
                st.write(f"• {search_data['query_text'][:60]}... ({search_data['results_count']} results)")

    except Exception as e:
        logger.error(f"Failed to display search history: {e}")


# Import time for search timing
import time