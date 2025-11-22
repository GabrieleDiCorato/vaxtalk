"""
PDF Handler for RAG System

This module provides functionality to load and process PDF documents
for use in a Retrieval-Augmented Generation (RAG) system.
"""

import os
from pypdf import PdfReader
from src.model.document_chunk import DocumentChunk, DocType


class PdfHandler:
    """
    Handles loading and chunking of PDF documents.

    This class provides methods to:
    - Chunk text into overlapping segments
    - Load and process PDF files from a folder
    """

    @staticmethod
    def chunk_text(text: str,
                   source: str,
                   doc_type: DocType,
                   chunk_size: int = 800,
                   chunk_overlap: int = 200,
                   start_id: int = 0) -> list[DocumentChunk]:
        """
        Splits text into overlapping chunks for RAG.

        Args:
            text: The text to chunk
            source: Source identifier (URL or file path)
            doc_type: Type of document (DocType.WEB or DocType.PDF)
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
            start_id: Starting ID for chunks

        Returns:
            List of DocumentChunk objects
        """
        words = text.split()
        chunks = []
        i = 0
        current_id = start_id

        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        id=current_id,
                        content=chunk_text,
                        source=source,
                        doc_type=doc_type
                    )
                )
                current_id += 1
            i += chunk_size - chunk_overlap

        return chunks

    @staticmethod
    def load_pdfs_from_folder(folder_path: str,
                              chunk_size: int = 800,
                              chunk_overlap: int = 200) -> list[DocumentChunk]:
        """
        Reads all PDFs in a folder and converts them to chunks.

        Args:
            folder_path: Path to folder containing PDF files
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks

        Returns:
            List of DocumentChunk objects from all PDFs
        """
        all_chunks: list[DocumentChunk] = []
        current_id = 0

        if not os.path.exists(folder_path):
            print(f"[PDF] Folder not found: {folder_path}")
            return []

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(folder_path, filename)
            print(f"[PDF] Loading: {full_path}")

            try:
                reader = PdfReader(full_path)
                text_pages = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text_pages.append(page_text)

                full_text = "\n".join(text_pages)
                chunks = PdfHandler.chunk_text(
                    text=full_text,
                    source=full_path,
                    doc_type=DocType.PDF,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    start_id=current_id
                )
                all_chunks.extend(chunks)
                current_id = all_chunks[-1].id + 1 if all_chunks else current_id

            except Exception as e:
                print(f"[PDF] ERROR on {full_path}: {e}")

        print(f"[PDF] Total chunks from PDFs: {len(all_chunks)}")
        return all_chunks
