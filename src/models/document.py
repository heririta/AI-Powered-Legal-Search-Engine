"""
Legal document model for the AI-Powered Legal Search Engine.

This module defines the LegalDocument entity and related data structures.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ProcessingStatus(Enum):
    """Processing status for documents."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LegalDocument:
    """Represents an uploaded legal document (UU/Statute) with metadata."""

    # Primary fields
    id: Optional[int] = None
    filename: str = ""
    file_path: str = ""
    file_size: int = 0
    file_type: str = ""
    uu_title: str = ""

    # Processing metadata
    upload_date: Optional[datetime] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: Optional[str] = None
    chunk_count: int = 0

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.upload_date is None:
            self.upload_date = datetime.now()

        if self.created_at is None:
            self.created_at = datetime.now()

        if self.updated_at is None:
            self.updated_at = datetime.now()

        # Convert string status to enum if needed
        if isinstance(self.processing_status, str):
            self.processing_status = ProcessingStatus(self.processing_status)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        data = asdict(self)

        # Convert enums to strings
        if isinstance(self.processing_status, ProcessingStatus):
            data['processing_status'] = self.processing_status.value

        # Convert datetime objects to strings
        if self.upload_date:
            data['upload_date'] = self.upload_date.isoformat()
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LegalDocument':
        """Create from dictionary."""
        # Handle datetime fields
        if 'upload_date' in data and data['upload_date']:
            if isinstance(data['upload_date'], str):
                data['upload_date'] = datetime.fromisoformat(data['upload_date'])

        if 'created_at' in data and data['created_at']:
            if isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])

        if 'updated_at' in data and data['updated_at']:
            if isinstance(data['updated_at'], str):
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        return cls(**data)

    def validate(self) -> bool:
        """Validate document data."""
        errors = []

        # Required fields
        if not self.filename or not self.filename.strip():
            errors.append("Filename is required")

        if not self.file_path or not self.file_path.strip():
            errors.append("File path is required")

        if not self.file_type:
            errors.append("File type is required")

        if not self.uu_title or not self.uu_title.strip():
            errors.append("UU title is required")

        # File size validation
        if self.file_size <= 0:
            errors.append("File size must be positive")

        if self.file_size > 50 * 1024 * 1024:  # 50MB limit
            errors.append("File size exceeds maximum limit (50MB)")

        # File type validation
        supported_types = ['pdf', 'txt']
        if self.file_type.lower() not in supported_types:
            errors.append(f"Unsupported file type: {self.file_type}")

        # Chunk count validation
        if self.chunk_count < 0:
            errors.append("Chunk count cannot be negative")

        if errors:
            logger.error(f"Document validation failed: {', '.join(errors)}")
            return False

        return True

    def update_status(self, status: ProcessingStatus, error_message: Optional[str] = None, chunk_count: Optional[int] = None) -> None:
        """Update processing status."""
        self.processing_status = status
        self.updated_at = datetime.now()

        if error_message:
            self.processing_error = error_message

        if chunk_count is not None:
            self.chunk_count = chunk_count

        if status == ProcessingStatus.COMPLETED:
            self.processing_error = None

    def is_processed(self) -> bool:
        """Check if document is successfully processed."""
        return self.processing_status == ProcessingStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if document processing failed."""
        return self.processing_status == ProcessingStatus.FAILED

    def is_processing(self) -> bool:
        """Check if document is currently being processed."""
        return self.processing_status == ProcessingStatus.PROCESSING

    def get_file_size_mb(self) -> float:
        """Get file size in MB."""
        return self.file_size / (1024 * 1024)

    def get_status_display(self) -> str:
        """Get user-friendly status display."""
        status_map = {
            ProcessingStatus.PENDING: "⏳ Pending",
            ProcessingStatus.PROCESSING: "🔄 Processing",
            ProcessingStatus.COMPLETED: "✅ Completed",
            ProcessingStatus.FAILED: "❌ Failed"
        }
        return status_map.get(self.processing_status, "❓ Unknown")

    def get_progress_info(self) -> Dict[str, Any]:
        """Get progress information for UI display."""
        return {
            "status": self.processing_status.value,
            "status_display": self.get_status_display(),
            "chunks_processed": self.chunk_count,
            "error_message": self.processing_error,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class ProcessingStatusInfo:
    """Processing status information for UI."""
    document_id: int
    status: str
    progress: float  # 0.0 to 1.0
    chunks_processed: int
    total_chunks: int
    error_message: Optional[str]
    estimated_time_remaining: Optional[int]

    def get_progress_percentage(self) -> int:
        """Get progress as percentage."""
        return int(self.progress * 100)

    def get_progress_bar(self) -> str:
        """Get text progress bar."""
        filled = int(self.progress * 20)
        bar = "█" * filled + "░" * (20 - filled)
        return f"[{bar}] {self.get_progress_percentage()}%"

    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.status == ProcessingStatus.COMPLETED.value

    def has_error(self) -> bool:
        """Check if processing has error."""
        return self.status == ProcessingStatus.FAILED.value or self.error_message is not None


def create_document_from_upload(filename: str, file_path: str, file_size: int, file_type: str, uu_title: str) -> LegalDocument:
    """Create a new LegalDocument from upload information."""
    return LegalDocument(
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        file_type=file_type.lower(),
        uu_title=uu_title,
        upload_date=datetime.now(),
        processing_status=ProcessingStatus.PENDING
    )


# Import logger after avoiding circular import
import logging
logger = logging.getLogger(__name__)