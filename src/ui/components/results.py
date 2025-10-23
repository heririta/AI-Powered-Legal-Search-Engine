"""
Search results display component for the AI-Powered Legal Search Engine.

This module displays search results with legal citations and provides
options for AI-powered question answering.
"""

import streamlit as st
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def highlight_keywords(text: str, query: str) -> str:
    """Highlight keywords from the search query in the text."""
    if not text or not query:
        return text

    # Extract keywords from the query (simple approach)
    # Remove common Indonesian/English stop words and punctuation
    stop_words = {
        'yang', 'dan', 'di', 'ke', 'pada', 'untuk', 'dari', 'dengan', 'adalah', 'ini', 'itu',
        'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'a', 'an'
    }

    # Split query into words and filter stop words
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word.lower() not in stop_words and len(word) > 2]

    if not keywords:
        return text

    # Create highlighting function with neutral theme colors
    def highlight_match(match):
        word = match.group()
        return f'<mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 3px; font-weight: 600;">{word}</mark>'

    # Highlight each keyword in the text (case-insensitive)
    highlighted_text = text
    for keyword in keywords:
        pattern = re.compile(r'\b(' + re.escape(keyword) + r')\b', re.IGNORECASE)
        highlighted_text = pattern.sub(highlight_match, highlighted_text)

    return highlighted_text


def render_search_results(query: str, results: List[Dict], search_time_ms: float):
    """Render search results with AI Q&A options."""
    # Search summary
    st.success(f"🎉 Found {len(results)} relevant results in {search_time_ms:.0f}ms")

    # Display query
    st.markdown(f"**🔍 Query:** {query}")

    # AI Q&A option
    st.markdown("---")
    if st.button("🤖 Get AI-Powered Answer with Citations", type="primary", key="ai_answer"):
        generate_ai_answer(query, results)

    # Results section
    st.markdown("### 📋 Search Results")

    for i, result in enumerate(results):
        chunk = result['chunk']
        similarity = result['similarity_score']
        rank = result['rank']

        with st.expander(
            f"**{rank}.** {chunk['pasal']} {chunk['ayat']} - {chunk['uu_title']} "
            f"(Similarity: {similarity:.2%})",
            expanded=(i == 0)  # Expand first result by default
        ):
            display_search_result(chunk, similarity, rank, query)


def display_search_result(chunk: Dict[str, Any], similarity: float, rank: int, query: str):
    """Display individual search result."""
    # Citation information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**📚 UU Title:** {chunk['uu_title']}")

    with col2:
        if chunk['bab']:
            st.write(f"**📖 Chapter:** {chunk['bab']}")

    with col3:
        st.write(f"**🏆 Rank:** {rank}")

    # Legal citation
    st.markdown(f"**⚖️ Legal Citation:** {chunk['pasal']} {chunk['ayat']}" +
                (f" {chunk['butir']}" if chunk['butir'] else ""))

    # Similarity score with visual indicator
    similarity_color = get_similarity_color(similarity)
    st.markdown(
        f"**🎯 Relevance:** "
        f"<span style='color: {similarity_color}; font-weight: bold;'>{similarity:.2%}</span>",
        unsafe_allow_html=True
    )

    # Full text with keyword highlighting
    st.markdown("**📝 Full Text:**")
    highlighted_text = highlight_keywords(chunk['ayat_text'], query)
    st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(f"📋 Copy Citation", key=f"copy_{rank}_{chunk['id']}_{hash(chunk['ayat_text']) % 10000}"):
            citation = format_citation(chunk)
            st.code(citation, language=None)
            st.success("Citation copied to clipboard! (In a real implementation)")

    with col2:
        if st.button(f"🔍 Find Similar", key=f"similar_{rank}_{chunk['id']}_{hash(chunk['ayat_text']) % 10000}"):
            st.info("Would search for similar clauses... (Feature not implemented)")

    with col3:
        if st.button(f"💾 Save Result", key=f"save_{rank}_{chunk['id']}_{hash(chunk['ayat_text']) % 10000}"):
            st.success("Result saved! (Feature not implemented)")


def generate_ai_answer(query: str, results: List[Dict]):
    """Generate AI-powered answer with citations."""
    try:
        with st.spinner("🤖 Generating AI answer..."):
            # For now, simulate RAG response
            # In a real implementation, this would:
            # 1. Assemble context from search results
            # 2. Send to Groq LLM API
            # 3. Extract and format citations
            # 4. Display the response

            ai_response = simulate_rag_response(query, results)

            # Display AI response
            st.markdown("---")
            st.markdown("### 🤖 AI-Powered Answer")

            # Answer content
            st.markdown(ai_response['answer'])

            # Citations
            if ai_response['citations']:
                st.markdown("#### 📚 Legal Citations")
                for i, citation in enumerate(ai_response['citations'], 1):
                    st.markdown(f"{i}. {citation}")

            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**🤖 Model:** {ai_response['model_used']}")
            with col2:
                st.write(f"⏱️ **Time:** {ai_response['generation_time_ms']}ms")
            with col3:
                st.write(f"🔤 **Sources:** {len(ai_response['sources'])} chunks")

            # Disclaimer
            st.markdown("---")
            st.warning(
                "⚠️ **Disclaimer:** This AI-generated answer is for informational purposes only "
                "and should not be considered legal advice. Please consult with a qualified "
                "legal professional for specific legal matters."
            )

            # Feedback
            st.markdown("### 💬 Feedback")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful", key="feedback_positive"):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 Not Helpful", key="feedback_negative"):
                    st.info("Thank you for your feedback - this helps us improve!")

    except Exception as e:
        st.error(f"❌ Failed to generate AI answer: {e}")
        logger.error(f"RAG response generation failed: {e}")


def simulate_rag_response(query: str, results: List[Dict]) -> Dict[str, Any]:
    """
    Simulate RAG response for demonstration.

    In a real implementation, this would:
    1. Assemble context from the retrieved chunks
    2. Send query + context to Groq LLM
    3. Parse response and extract citations
    4. Return structured response
    """
    # Extract context chunks
    context_chunks = []
    citations = []

    for result in results[:3]:  # Use top 3 results for context
        chunk = result['chunk']
        context_chunks.append(chunk['ayat_text'])

        citation = format_citation(chunk)
        citations.append(citation)

    # Simulate AI response based on query
    if "investment" in query.lower() or "perusahaan" in query.lower():
        answer = f"""
        Based on the retrieved legal documents, here are the key requirements for establishing a company in Indonesia:

        **General Requirements:**
        - All citizens must comply with applicable laws and regulations (Pasal 1 Ayat (1))
        - The government must administer the state based on law and justice principles (Pasal 2 Ayat (1))

        **Recommendations:**
        1. Consult with a legal professional for specific company formation requirements
        2. Prepare necessary documentation for company registration
        3. Ensure compliance with relevant sector-specific regulations
        4. Consider investment structure and capital requirements

        **Next Steps:**
        - Determine the appropriate company type (PT, CV, Firma, etc.)
        - Prepare Articles of Association
        - Obtain necessary permits and licenses
        - Register with relevant government authorities
        """
    elif "pajak" in query.lower() or "tax" in query.lower():
        answer = f"""
        Based on Indonesian tax law, here are the key tax obligations:

        **Tax Compliance Requirements:**
        - Every citizen has an obligation to comply with applicable tax laws (Pasal 1 Ayat (2))
        - The government must administer the state based on law and justice principles (Pasal 2 Ayat (1))

        **Key Points:**
        1. All individuals and entities must fulfill tax obligations
        2. Tax compliance is mandatory for legal entities
        3. The government ensures fair and just tax administration

        **Recommendations:**
        - Consult with a tax professional for specific tax advice
        - Maintain proper tax records and documentation
        - File tax returns on time
        - Stay updated on tax regulations
        """
    else:
        answer = f"""
        Based on the retrieved legal documents, here's what I found regarding your query:

        **Legal Framework:**
        The Indonesian legal system establishes that all citizens must comply with applicable laws and regulations (Pasal 1 Ayat (2)). Additionally, the government must administer the state based on law and justice principles (Pasal 2 Ayat (1)).

        **Key Principles:**
        1. Legal compliance is mandatory for all citizens
        2. Government administration must be based on law
        3. Justice principles must be upheld in governance

        **For Specific Guidance:**
        I recommend consulting with a qualified legal professional who can provide advice tailored to your specific situation and needs.
        """

    return {
        'answer': answer,
        'citations': citations[:3],  # Show top 3 citations
        'sources': results[:3],
        'model_used': 'mixtral-8x7b-32768',
        'generation_time_ms': 1500,
        'tokens_used': 450
    }


def format_citation(chunk: Dict[str, Any]) -> str:
    """Format legal citation from chunk data."""
    parts = [chunk['uu_title']]

    if chunk['bab']:
        parts.append(chunk['bab'])

    parts.append(chunk['pasal'])
    parts.append(chunk['ayat'])

    if chunk['butir']:
        parts.append(chunk['butir'])

    return " ".join(parts)


def get_similarity_color(similarity: float) -> str:
    """Get color for similarity score based on threshold using neutral theme."""
    if similarity >= 0.6:
        return "#2e7d32"  # Dark green - neutral but positive
    elif similarity >= 0.4:
        return "#f57c00"  # Dark orange - neutral but cautionary
    else:
        return "#c62828"  # Dark red - neutral but concerning


def display_no_results_message():
    """Display message when no results are found."""
    st.warning("🔍 No results found matching your query.")

    st.markdown("### 💡 Suggestions:")
    st.markdown("""
    - Try different keywords or phrases
    - Check spelling and grammar
    - Use more general terms
    - Ensure documents are uploaded and processed
    - Adjust search filters if applied
    """)

    st.markdown("### 📚 Search Examples:")
    examples = [
        "What are the requirements for business registration?",
        "How to comply with tax regulations?",
        "What are labor rights in Indonesia?",
        "Apa saja ketentuan investasi asing?",
    ]

    for example in examples:
        if st.button(f"💭 {example}", key=f"suggestion_{example[:20]}"):
            st.session_state.search_query = example
            st.rerun()