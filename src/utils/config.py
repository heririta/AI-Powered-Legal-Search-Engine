"""
Configuration management for the AI-Powered Legal Search Engine.

This module handles loading and validating configuration from environment variables
and provides defaults for all required settings.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AppConfig(BaseModel):
    """Application configuration with validation."""

    # Database Configuration
    database_path: str = Field(default="src/data/.legal_db.sqlite", description="Path to SQLite database")

    # Vector Store Configuration
    faiss_index_path: str = Field(default="src/data/.faiss_index", description="Path to FAISS index file")
    embedding_dimension: int = Field(default=1024, description="Embedding vector dimension")

    # File Upload Configuration
    upload_dir: str = Field(default="src/data/uploads", description="Upload directory for documents")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    supported_formats: List[str] = Field(default=["pdf", "txt"], description="Supported file formats")

    # API Configuration
    cohere_model: str = Field(default="embed-multilingual-v3.0", description="Cohere embedding model")
    groq_model: str = Field(default="mixtral-8x7b-32768", description="Groq LLM model")
    embedding_batch_size: int = Field(default=100, description="Batch size for embedding generation")
    max_search_results: int = Field(default=20, description="Maximum search results to return")

    # Performance Configuration
    search_timeout_ms: int = Field(default=5000, description="Search timeout in milliseconds")
    llm_timeout_ms: int = Field(default=30000, description="LLM timeout in milliseconds")
    max_concurrent_processes: int = Field(default=1, description="Maximum concurrent document processes")

    # UI Configuration
    page_size: int = Field(default=10, description="Number of results per page")
    max_query_length: int = Field(default=500, description="Maximum query length in characters")
    result_highlight_threshold: float = Field(default=0.4, description="Similarity threshold for result highlighting")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/application.log", description="Log file path")

    # API Keys (required)
    cohere_api_key: str = Field(..., description="Cohere API key")
    groq_api_key: str = Field(..., description="Groq API key")

    @validator('database_path', 'faiss_index_path', 'upload_dir', 'log_file')
    def ensure_directory_exists(cls, v):
        """Ensure directory exists for file paths."""
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v

    @validator('supported_formats')
    def validate_formats(cls, v):
        """Validate supported file formats."""
        if not v:
            raise ValueError("At least one supported format must be specified")
        return [fmt.lower().strip() for fmt in v if fmt.strip()]

    @validator('max_file_size_mb')
    def validate_file_size(cls, v):
        """Validate maximum file size."""
        if v <= 0:
            raise ValueError("File size must be positive")
        if v > 500:  # Reasonable upper limit
            raise ValueError("File size too large (max 500MB)")
        return v

    @validator('embedding_dimension')
    def validate_embedding_dimension(cls, v):
        """Validate embedding dimension."""
        if v <= 0:
            raise ValueError("Embedding dimension must be positive")
        return v

    @validator('embedding_batch_size')
    def validate_batch_size(cls, v):
        """Validate embedding batch size."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        if v > 1000:  # Reasonable upper limit for API calls
            raise ValueError("Batch size too large (max 1000)")
        return v

    @validator('max_search_results')
    def validate_max_results(cls, v):
        """Validate maximum search results."""
        if v <= 0:
            raise ValueError("Max results must be positive")
        if v > 100:  # Reasonable upper limit
            raise ValueError("Max results too large (max 100)")
        return v

    @validator('search_timeout_ms', 'llm_timeout_ms')
    def validate_timeouts(cls, v):
        """Validate timeout values."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @validator('result_highlight_threshold')
    def validate_threshold(cls, v):
        """Validate similarity threshold."""
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        return cls(
            # Database
            database_path=os.getenv('DATABASE_PATH', 'src/data/.legal_db.sqlite'),
            faiss_index_path=os.getenv('FAISS_INDEX_PATH', 'src/data/.faiss_index'),
            embedding_dimension=int(os.getenv('EMBEDDING_DIMENSION', '1024')),

            # File Upload
            upload_dir=os.getenv('UPLOAD_DIR', 'src/data/uploads'),
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', '100')),
            supported_formats=os.getenv('SUPPORTED_FORMATS', 'pdf,txt').split(','),

            # API
            cohere_model=os.getenv('COHERE_MODEL', 'embed-multilingual-v3.0'),
            groq_model=os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768'),
            embedding_batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '100')),
            max_search_results=int(os.getenv('MAX_SEARCH_RESULTS', '20')),

            # Performance
            search_timeout_ms=int(os.getenv('SEARCH_TIMEOUT_MS', '5000')),
            llm_timeout_ms=int(os.getenv('LLM_TIMEOUT_MS', '30000')),
            max_concurrent_processes=int(os.getenv('MAX_CONCURRENT_PROCESSES', '1')),

            # UI
            page_size=int(os.getenv('PAGE_SIZE', '10')),
            max_query_length=int(os.getenv('MAX_QUERY_LENGTH', '500')),
            result_highlight_threshold=float(os.getenv('RESULT_HIGHLIGHT_THRESHOLD', '0.4')),

            # Logging
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'logs/application.log'),

            # API Keys
            cohere_api_key=os.getenv('COHERE_API_KEY', ''),
            groq_api_key=os.getenv('GROQ_API_KEY', ''),
        )

    def validate_api_keys(self) -> bool:
        """Validate that required API keys are present."""
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")
        return True


# Global configuration instance
config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = AppConfig.from_env()
        config.validate_api_keys()
    return config


def reload_config() -> AppConfig:
    """Reload configuration from environment variables."""
    global config
    config = AppConfig.from_env()
    config.validate_api_keys()
    return config