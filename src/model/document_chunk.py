from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path

from pydantic_core import Url

class DocType(str, Enum):
    WEB = "web"
    PDF = "pdf"


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document with metadata."""

    id: int = Field(..., description="Unique identifier for the document chunk")
    content: str = Field(..., description="Text content of the document chunk")
    source: str | Url | Path = Field(..., description="URL or file path of the document source")
    doc_type: DocType = Field(..., description="Type of the document, e.g., 'web' or 'pdf'")
