"""
Database service for the AI-Powered Legal Search Engine.

This module handles SQLite database operations, schema creation, and migrations.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from utils.config import get_config

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for SQLite database operations."""

    def __init__(self):
        """Initialize database service."""
        self.config = get_config()
        self.db_path = self.config.database_path

    def initialize_database(self) -> bool:
        """Create database schema and initialize tables."""
        try:
            with self.get_connection() as conn:
                # Create tables
                self._create_documents_table(conn)
                self._create_chunks_table(conn)
                self._create_search_queries_table(conn)
                self._create_search_results_table(conn)
                self._create_ai_responses_table(conn)

                # Create indexes
                self._create_indexes(conn)

                conn.commit()
                logger.info(f"Database initialized successfully at {self.db_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    def _create_documents_table(self, conn: sqlite3.Connection) -> None:
        """Create legal_documents table."""
        query = """
        CREATE TABLE IF NOT EXISTS legal_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename VARCHAR(255) NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size INTEGER NOT NULL,
            file_type VARCHAR(10) NOT NULL,
            uu_title VARCHAR(500) NOT NULL,
            upload_date DATETIME NOT NULL,
            processing_status VARCHAR(20) DEFAULT 'pending',
            processing_error TEXT,
            chunk_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        conn.execute(query)

    def _create_chunks_table(self, conn: sqlite3.Connection) -> None:
        """Create legal_chunks table."""
        query = """
        CREATE TABLE IF NOT EXISTS legal_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            embedding_id VARCHAR(100) NOT NULL UNIQUE,
            uu_title VARCHAR(500) NOT NULL,
            bab VARCHAR(200),
            pasal VARCHAR(100),
            ayat VARCHAR(100),
            butir VARCHAR(100),
            ayat_text TEXT NOT NULL,
            chunk_order INTEGER NOT NULL,
            embedding_vector BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES legal_documents (id) ON DELETE CASCADE
        )
        """
        conn.execute(query)

    def _create_search_queries_table(self, conn: sqlite3.Connection) -> None:
        """Create search_queries table."""
        query = """
        CREATE TABLE IF NOT EXISTS search_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text VARCHAR(500) NOT NULL,
            query_vector BLOB,
            filter_uu_title VARCHAR(500),
            filter_bab VARCHAR(200),
            search_type VARCHAR(20) DEFAULT 'semantic',
            results_count INTEGER DEFAULT 5,
            execution_time_ms INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        conn.execute(query)

    def _create_search_results_table(self, conn: sqlite3.Connection) -> None:
        """Create search_results table."""
        query = """
        CREATE TABLE IF NOT EXISTS search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_query_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            similarity_score REAL NOT NULL,
            rank INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (search_query_id) REFERENCES search_queries (id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES legal_chunks (id) ON DELETE CASCADE
        )
        """
        conn.execute(query)

    def _create_ai_responses_table(self, conn: sqlite3.Connection) -> None:
        """Create ai_responses table."""
        query = """
        CREATE TABLE IF NOT EXISTS ai_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_query_id INTEGER NOT NULL UNIQUE,
            response_text TEXT NOT NULL,
            context_chunks TEXT,
            citations TEXT,
            model_used VARCHAR(50) NOT NULL,
            tokens_used INTEGER,
            generation_time_ms INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (search_query_id) REFERENCES search_queries (id) ON DELETE CASCADE
        )
        """
        conn.execute(query)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_filename ON legal_documents(filename)",
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON legal_documents(processing_status)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON legal_chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_id ON legal_chunks(embedding_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_legal_hierarchy ON legal_chunks(uu_title, pasal, ayat)",
            "CREATE INDEX IF NOT EXISTS idx_search_results_query_id ON search_results(search_query_id)",
            "CREATE INDEX IF NOT EXISTS idx_search_results_chunk_id ON search_results(chunk_id)",
            "CREATE INDEX IF NOT EXISTS idx_search_results_rank ON search_results(search_query_id, rank)",
        ]

        for index_query in indexes:
            conn.execute(index_query)

    @contextmanager
    def get_connection(self):
        """Get database connection with context management."""
        conn = None
        try:
            # Ensure database directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows

            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")

            yield conn

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def save_document(self, document_data: Dict[str, Any]) -> Optional[int]:
        """Save document metadata and return document ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO legal_documents
                    (filename, file_path, file_size, file_type, uu_title, upload_date, processing_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_data['filename'],
                    document_data['file_path'],
                    document_data['file_size'],
                    document_data['file_type'],
                    document_data['uu_title'],
                    document_data['upload_date'],
                    document_data.get('processing_status', 'pending')
                ))

                document_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Document saved with ID: {document_id}")
                return document_id

        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            return None

    def update_document_status(self, document_id: int, status: str, error_message: Optional[str] = None, chunk_count: Optional[int] = None) -> bool:
        """Update document processing status."""
        try:
            with self.get_connection() as conn:
                update_fields = ["processing_status = ?", "updated_at = CURRENT_TIMESTAMP"]
                params = [status]

                if error_message:
                    update_fields.append("processing_error = ?")
                    params.append(error_message)

                if chunk_count is not None:
                    update_fields.append("chunk_count = ?")
                    params.append(chunk_count)

                params.append(document_id)

                query = f"UPDATE legal_documents SET {', '.join(update_fields)} WHERE id = ?"
                conn.execute(query, params)
                conn.commit()

                logger.info(f"Document {document_id} status updated to: {status}")
                return True

        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            return False

    def get_document_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM legal_documents WHERE id = ?",
                    (document_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None

    def save_chunks(self, chunks_data: List[Dict[str, Any]]) -> bool:
        """Save multiple legal chunks."""
        try:
            with self.get_connection() as conn:
                for chunk in chunks_data:
                    conn.execute("""
                        INSERT OR REPLACE INTO legal_chunks
                        (document_id, embedding_id, uu_title, bab, pasal, ayat, butir, ayat_text, chunk_order, embedding_vector)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk['document_id'],
                        chunk['embedding_id'],
                        chunk['uu_title'],
                        chunk.get('bab'),
                        chunk.get('pasal'),
                        chunk.get('ayat'),
                        chunk.get('butir'),
                        chunk['ayat_text'],
                        chunk['chunk_order'],
                        chunk.get('embedding_vector')
                    ))

                conn.commit()
                logger.info(f"Saved {len(chunks_data)} chunks")
                return True

        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            return False

    def get_chunks_by_document(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM legal_chunks WHERE document_id = ? ORDER BY chunk_order",
                    (document_id,)
                )
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get chunks: {e}")
            return []

    def get_chunk_by_embedding_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by embedding ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM legal_chunks WHERE embedding_id = ?",
                    (embedding_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None

        except Exception as e:
            logger.error(f"Failed to get chunk: {e}")
            return None

    def get_legal_citations(self, uu_title: str, pasal: str, ayat: str) -> List[Dict[str, Any]]:
        """Get specific legal citations."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM legal_chunks
                    WHERE uu_title = ? AND pasal = ? AND ayat = ?
                    ORDER BY document_id, chunk_order
                """, (uu_title, pasal, ayat))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get legal citations: {e}")
            return []

    def save_search_query(self, query_data: Dict[str, Any]) -> Optional[int]:
        """Save search query."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO search_queries
                    (query_text, query_vector, filter_uu_title, filter_bab, search_type, results_count, execution_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_data['query_text'],
                    query_data.get('query_vector'),
                    query_data.get('filter_uu_title'),
                    query_data.get('filter_bab'),
                    query_data.get('search_type', 'semantic'),
                    query_data.get('results_count', 5),
                    query_data.get('execution_time_ms')
                ))

                query_id = cursor.lastrowid
                conn.commit()

                return query_id

        except Exception as e:
            logger.error(f"Failed to save search query: {e}")
            return None

    def save_search_results(self, results_data: List[Dict[str, Any]]) -> bool:
        """Save search results."""
        try:
            with self.get_connection() as conn:
                for result in results_data:
                    conn.execute("""
                        INSERT INTO search_results
                        (search_query_id, chunk_id, similarity_score, rank)
                        VALUES (?, ?, ?, ?)
                    """, (
                        result['search_query_id'],
                        result['chunk_id'],
                        result['similarity_score'],
                        result['rank']
                    ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to save search results: {e}")
            return False

    def save_ai_response(self, response_data: Dict[str, Any]) -> bool:
        """Save AI response."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ai_responses
                    (search_query_id, response_text, context_chunks, citations, model_used, tokens_used, generation_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    response_data['search_query_id'],
                    response_data['response_text'],
                    response_data.get('context_chunks'),
                    response_data.get('citations'),
                    response_data['model_used'],
                    response_data.get('tokens_used'),
                    response_data.get('generation_time_ms')
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to save AI response: {e}")
            return False

    def delete_document(self, document_id: int) -> bool:
        """Delete a document and all its associated data.

        Due to foreign key constraints with ON DELETE CASCADE:
        - Deleting a document will automatically delete its chunks
        - Deleting chunks will automatically delete search results
        """
        try:
            with self.get_connection() as conn:
                # First, get the file path to delete the actual file
                cursor = conn.execute(
                    "SELECT file_path FROM legal_documents WHERE id = ?",
                    (document_id,)
                )
                doc = cursor.fetchone()

                if not doc:
                    logger.warning(f"Document with ID {document_id} not found")
                    return False

                file_path = doc['file_path']

                # Delete the document (cascading will handle chunks and search results)
                cursor = conn.execute(
                    "DELETE FROM legal_documents WHERE id = ?",
                    (document_id,)
                )

                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Successfully deleted document {document_id} and all associated data")

                    # Attempt to delete the actual file
                    try:
                        import os
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted file: {file_path}")
                    except Exception as file_error:
                        logger.warning(f"Failed to delete file {file_path}: {file_error}")

                    return True
                else:
                    logger.warning(f"No document found with ID {document_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def get_document_chunk_embedding_ids(self, document_id: int) -> List[str]:
        """Get all embedding IDs for chunks belonging to a document."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT embedding_id FROM legal_chunks WHERE document_id = ? AND embedding_id IS NOT NULL",
                    (document_id,)
                )
                return [row['embedding_id'] for row in cursor.fetchall() if row['embedding_id']]
        except Exception as e:
            logger.error(f"Failed to get document embedding IDs: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.get_connection() as conn:
                stats = {}

                # Document count
                cursor = conn.execute("SELECT COUNT(*) as count FROM legal_documents")
                stats['documents_count'] = cursor.fetchone()['count']

                # Chunks count
                cursor = conn.execute("SELECT COUNT(*) as count FROM legal_chunks")
                stats['chunks_count'] = cursor.fetchone()['count']

                # Search queries count
                cursor = conn.execute("SELECT COUNT(*) as count FROM search_queries")
                stats['search_queries_count'] = cursor.fetchone()['count']

                # AI responses count
                cursor = conn.execute("SELECT COUNT(*) as count FROM ai_responses")
                stats['ai_responses_count'] = cursor.fetchone()['count']

                # Database file size
                if Path(self.db_path).exists():
                    stats['database_size_mb'] = Path(self.db_path).stat().st_size / (1024 * 1024)
                else:
                    stats['database_size_mb'] = 0

                return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}


# Global database service instance
db_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """Get the global database service instance."""
    global db_service
    if db_service is None:
        db_service = DatabaseService()
    return db_service