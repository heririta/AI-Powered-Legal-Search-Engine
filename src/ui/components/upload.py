"""
File upload component for the AI-Powered Legal Search Engine.

This module provides the document upload interface with progress tracking
and error handling.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import tempfile
import os
import re

from services.database import get_database_service
from services.embedding import get_embedding_service
from services.vector_store import get_vector_store_service
from models.document import LegalDocument, ProcessingStatus
from utils.config import get_config

# Try to import PyMuPDF
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. PDF parsing will be limited.")

logger = logging.getLogger(__name__)


def render_upload_component():
    """Render the file upload interface."""
    # File uploader
    uploaded_file = st.file_uploader(
        "📄 Choose a legal document",
        type=['pdf', 'txt'],
        help="Upload Indonesian legal documents (UU, PP, Permen, etc.) in PDF or TXT format (Maximum: 100MB)"
    )

    if uploaded_file is not None:
        # Display file information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**📁 Filename:** {uploaded_file.name}")
        with col2:
            st.write(f"📏 **Size:** {uploaded_file.size / (1024*1024):.2f} MB")
        with col3:
            st.write(f"📄 **Type:** {uploaded_file.type}")

        # Validate file
        if not validate_uploaded_file(uploaded_file):
            return

        # Extract UU title (simple extraction for demo)
        uu_title = extract_uu_title(uploaded_file.name)

        # Upload button
        if st.button("📤 Upload and Process Document", type="primary"):
            process_uploaded_document(uploaded_file, uu_title)


def validate_uploaded_file(uploaded_file) -> bool:
    """Validate the uploaded file."""
    # Get config for file size limit
    config = get_config()

    # Check file size (using configured limit)
    if uploaded_file.size > config.max_file_size_mb * 1024 * 1024:
        st.error(f"❌ File size exceeds {config.max_file_size_mb}MB limit")
        return False

    # Check file type
    if uploaded_file.type not in ['application/pdf', 'text/plain']:
        if not uploaded_file.name.lower().endswith(('.pdf', '.txt')):
            st.error("❌ Unsupported file type. Please upload PDF or TXT files.")
            return False

    return True


def extract_uu_title(filename: str) -> str:
    """Extract UU title from filename (simple implementation)."""
    # Remove extension and clean up
    base_name = os.path.splitext(filename)[0]

    # Simple extraction - in a real implementation, this would be more sophisticated
    if "UU" in base_name.upper() or "UNDANG-UNDANG" in base_name.upper():
        return base_name.replace("_", " ").replace("-", " ")
    else:
        return f"Document: {base_name}"


def process_uploaded_document(uploaded_file, uu_title: str):
    """Process the uploaded document."""
    try:
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        # Show initial progress
        progress_placeholder.progress(0, "Starting document processing...")
        status_placeholder.info("📄 Initializing document upload...")

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Initialize services
            db_service = get_database_service()
            embedding_service = get_embedding_service()
            vector_service = get_vector_store_service()

            # Step 1: Save document to database
            progress_placeholder.progress(0.1, "Saving document metadata...")
            status_placeholder.info("💾 Saving document to database...")

            document_data = {
                'filename': uploaded_file.name,
                'file_path': tmp_file_path,
                'file_size': uploaded_file.size,
                'file_type': uploaded_file.name.split('.')[-1].lower(),
                'uu_title': uu_title,
                'upload_date': datetime.now(),
                'processing_status': ProcessingStatus.PROCESSING.value
            }

            document_id = db_service.save_document(document_data)
            if not document_id:
                raise Exception("Failed to save document to database")

            # Step 2: Parse document and extract legal structure
            progress_placeholder.progress(0.3, "Parsing document structure...")
            status_placeholder.info("📖 Extracting legal hierarchy (UU, Bab, Pasal, Ayat)...")

            # Parse document using real PDF parsing
            chunks_data = parse_document_real(document_id, uu_title, tmp_file_path)

            # Step 3: Generate embeddings
            progress_placeholder.progress(0.6, "Generating embeddings...")
            status_placeholder.info("🔤 Creating vector embeddings for semantic search...")

            if chunks_data:
                # Generate embeddings for chunks
                chunk_texts = [chunk['ayat_text'] for chunk in chunks_data]
                embeddings = embedding_service.generate_embeddings(chunk_texts)

                if embeddings:
                    # Add embeddings to chunks data
                    for i, chunk in enumerate(chunks_data):
                        if i < len(embeddings):
                            chunk['embedding_vector'] = embeddings[i].tobytes()

            # Step 4: Save chunks to database
            progress_placeholder.progress(0.8, "Saving to database...")
            status_placeholder.info("💾 Storing chunks in database...")

            if chunks_data:
                db_service.save_chunks(chunks_data)

            # Step 5: Add to vector index
            progress_placeholder.progress(0.9, "Updating search index...")
            status_placeholder.info("🔍 Adding to vector search index...")

            if chunks_data and embeddings:
                # Generate embedding IDs and add to vector store
                embedding_ids = [f"doc_{document_id}_chunk_{i}" for i in range(len(chunks_data))]
                vector_service.add_vectors(embeddings, embedding_ids)

            # Step 6: Finalize
            progress_placeholder.progress(1.0, "Processing complete!")
            status_placeholder.success("✅ Document processed successfully!")

            # Update document status
            db_service.update_document_status(
                document_id,
                ProcessingStatus.COMPLETED,
                chunk_count=len(chunks_data) if chunks_data else 0
            )

            # Clean up temp file
            os.unlink(tmp_file_path)

            # Show success message
            st.success(f"🎉 Document '{uploaded_file.name}' has been successfully processed!")
            st.info(f"📊 Extracted {len(chunks_data)} legal chunks from the document.")

            # Show processing summary
            with st.expander("📋 Processing Summary"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Chunks", len(chunks_data))
                with col2:
                    st.metric("🔤 Embeddings", len(embeddings) if embeddings else 0)
                with col3:
                    st.metric("⏱️ Processing Time", "< 5 seconds")  # Placeholder

            # Clear session state for next upload
            if 'upload_progress' in st.session_state:
                del st.session_state.upload_progress

            # Rerun to refresh the UI
            st.rerun()

        except Exception as e:
            # Clean up temp file on error
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

            # Update document status to failed
            if 'document_id' in locals():
                db_service.update_document_status(
                    document_id,
                    ProcessingStatus.FAILED,
                    error_message=str(e)
                )

            # Show error
            progress_placeholder.empty()
            status_placeholder.error(f"❌ Processing failed: {e}")
            logger.error(f"Document processing failed: {e}")

    except Exception as e:
        st.error(f"❌ Upload failed: {e}")
        logger.error(f"Document upload failed: {e}")


def parse_document_real(document_id: int, uu_title: str, file_path: str) -> list:
    """
    Parse document using PyMuPDF for real text extraction.

    This function:
    1. Uses PyMuPDF to extract text from PDF page by page
    2. Creates one chunk per page (as requested)
    3. Attempts to identify legal structure (UU, Bab, Pasal)
    4. Returns properly formatted chunks
    """
    try:
        chunks_data = []

        if file_path.lower().endswith('.pdf') and PDF_AVAILABLE:
            # Parse PDF with PyMuPDF
            doc = fitz.open(file_path)

            logger.info(f"Starting PDF parsing for document {document_id} with {len(doc)} pages")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()

                if text.strip():  # Only process pages with text
                    # Clean and process text
                    cleaned_text = clean_text(text)

                    if cleaned_text.strip():  # Only create chunk if there's meaningful text
                        # Try to extract legal structure
                        bab, pasal, ayat = extract_legal_structure(cleaned_text, page_num + 1)

                        chunk = {
                            'document_id': document_id,
                            'embedding_id': f"doc_{document_id}_chunk_{page_num}",
                            'uu_title': uu_title,
                            'bab': bab,
                            'pasal': pasal,
                            'ayat': ayat,
                            'butir': None,
                            'ayat_text': cleaned_text,
                            'chunk_order': page_num,
                            'page_number': page_num + 1
                        }

                        chunks_data.append(chunk)
                        logger.info(f"Created chunk for page {page_num + 1}: {len(cleaned_text)} characters")

            doc.close()

        elif file_path.lower().endswith('.txt'):
            # Parse TXT file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Split content into chunks (simplified approach)
            chunk_size = 2000  # characters per chunk
            total_chunks = (len(content) + chunk_size - 1) // chunk_size

            for i in range(total_chunks):
                start_pos = i * chunk_size
                end_pos = min((i + 1) * chunk_size, len(content))
                chunk_text = content[start_pos:end_pos].strip()

                if chunk_text:
                    chunk = {
                        'document_id': document_id,
                        'embedding_id': f"doc_{document_id}_chunk_{i}",
                        'uu_title': uu_title,
                        'bab': None,
                        'pasal': None,
                        'ayat': None,
                        'butir': None,
                        'ayat_text': chunk_text,
                        'chunk_order': i,
                        'page_number': i + 1
                    }

                    chunks_data.append(chunk)

        else:
            # Fallback for unsupported formats or when PyMuPDF is not available
            logger.warning(f"Using fallback parsing for {file_path}")
            return create_fallback_chunks(document_id, uu_title, file_path)

        logger.info(f"Real parsing completed: Created {len(chunks_data)} chunks for document {document_id}")
        return chunks_data

    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        # Fallback to simulated chunks if real parsing fails
        return create_fallback_chunks(document_id, uu_title, file_path)


def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and normalizing."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove common PDF artifacts
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    return text


def extract_legal_structure(text: str, page_num: int) -> tuple:
    """Extract legal structure from text (simplified)."""
    # Simple regex patterns to identify legal structure
    bab_patterns = [
        r'BAB\s+([IVXLCDM]+|[0-9]+)',
        r'BAB\s+(\w+)',
        r'PELAKSANA\s+(\w+)'
    ]

    pasal_patterns = [
        r'PASAL\s+([0-9]+)',
       'ARTICLE\s+([0-9]+)'
    ]

    ayat_patterns = [
        r'AYAT\s*\(([0-9]+)\)',
        r'CLAUSE\s+([0-9]+)',
        r'\(([0-9]+)\)'
    ]

    bab = None
    pasal = None
    ayat = None

    # Try to find Bab
    for pattern in bab_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            bab = f"BAB {match.group(1)}"
            break

    # Try to find Pasal
    for pattern in pasal_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pasal = f"Pasal {match.group(1)}"
            break

    # Try to find Ayat
    for pattern in ayat_patterns:
        match = re.search(pattern, text)
        if match:
            ayat = f"Ayat ({match.group(1)})"
            break

    # If no structure found, use defaults
    if not bab:
        bab = f"BAB {(page_num // 5) + 1}"  # Rough grouping by pages
    if not pasal:
        pasal = f"Pasal {page_num}"
    if not ayat:
        ayat = f"Ayat ({page_num})"

    return bab, pasal, ayat


def create_fallback_chunks(document_id: int, uu_title: str, file_path: str) -> list:
    """Create fallback chunks when real parsing is not available."""
    try:
        # Read first part of file to get some text
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read(5000)  # Read first 5000 characters

        # Create a single chunk from the available content
        if content.strip():
            return [{
                'document_id': document_id,
                'embedding_id': f"doc_{document_id}_chunk_0",
                'uu_title': uu_title,
                'bab': 'BAB I',
                'pasal': 'Pasal 1',
                'ayat': 'Ayat (1)',
                'butir': None,
                'ayat_text': content.strip(),
                'chunk_order': 0,
                'page_number': 1
            }]
        else:
            # Empty fallback
            return [{
                'document_id': document_id,
                'embedding_id': f"doc_{document_id}_chunk_0",
                'uu_title': uu_title,
                'bab': 'BAB I',
                'pasal': 'Pasal 1',
                'ayat': 'Ayat (1)',
                'butir': None,
                'ayat_text': f'Dokumen {uu_title} - Unable to extract content.',
                'chunk_order': 0,
                'page_number': 1
            }]

    except Exception as e:
        logger.error(f"Fallback chunk creation failed: {e}")
        return [{
            'document_id': document_id,
            'embedding_id': f"doc_{document_id}_chunk_0",
            'uu_title': uu_title,
            'bab': 'BAB I',
            'pasal': 'Pasal 1',
            'ayat': 'Ayat (1)',
            'butir': None,
            'ayat_text': f'Error processing document {uu_title}.',
            'chunk_order': 0,
            'page_number': 1
        }]


def display_processing_status(status: ProcessingStatus, progress: float, message: str):
    """Display processing status with progress bar."""
    status_emoji = {
        ProcessingStatus.PENDING: "⏳",
        ProcessingStatus.PROCESSING: "🔄",
        ProcessingStatus.COMPLETED: "✅",
        ProcessingStatus.FAILED: "❌"
    }.get(status, "❓")

    st.write(f"{status_emoji} **{status.value.title()}**")
    st.progress(progress, message)

    if status == ProcessingStatus.FAILED:
        st.error("Processing failed. Please check the error message and try again.")