"""
RAG Knowledge Base Manager

This module provides functionality to build and query a knowledge base
for Retrieval-Augmented Generation (RAG) systems. It handles document
processing, embedding generation, and retrieval - without any ADK dependencies.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from src.model.document_chunk import DocumentChunk
from src.rag.pdf_handler import PdfHandler
from src.rag.web_handler import WebHandler
from src.rag.embedding_handler import EmbeddingHandler


class RagKnowledgeBase:
    """
    Manages the knowledge base for RAG systems.

    This class handles:
    - Document loading from PDFs and websites
    - Embedding generation and caching
    - Vector-based retrieval

    Does NOT include Agent/ADK functionality - that stays in the application layer.
    """

    def __init__(self, api_key: str, cache_dir: str | Path = "../cache"):
        """
        Initialize the knowledge base manager.

        Args:
            api_key: Google API key for embedding generation
            cache_dir: Directory to cache embeddings and chunks
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)

        # Initialize embedding handler
        self.embedding_handler = EmbeddingHandler(
            api_key=api_key,
            cache_dir=cache_dir
        )

        # Storage for embeddings and chunks
        self.embeddings: np.ndarray = np.array([])
        self.chunks: list[DocumentChunk] = []

    def build_knowledge_base(
        self,
        pdf_folder: Optional[str | Path] = None,
        root_url: Optional[str] = None,
        max_pages: int = 100,
        max_depth: int = 3,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        use_cache: bool = True
    ) -> None:
        """
        Build or load the knowledge base from documents.

        Args:
            pdf_folder: Path to folder containing PDF files
            root_url: Starting URL for web crawling
            max_pages: Maximum number of web pages to crawl
            max_depth: Maximum crawl depth from root URL
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
            use_cache: Whether to use cached embeddings if available
        """
        # Try to load from cache
        if use_cache:
            self.embeddings, self.chunks = self.embedding_handler.load_index_from_cache()

        # Build fresh if cache is empty
        if self.embeddings.size == 0:
            print("Building knowledge base...")

            all_chunks = []
            current_id = 0

            # Load PDFs if folder provided
            if pdf_folder:
                pdf_chunks = PdfHandler.load_pdfs_from_folder(
                    str(pdf_folder),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                all_chunks.extend(pdf_chunks)
                current_id = pdf_chunks[-1].id + 1 if pdf_chunks else 0

            # Crawl website if URL provided
            if root_url:
                web_chunks = WebHandler.crawl_website(
                    root_url=root_url,
                    max_pages=max_pages,
                    max_depth=max_depth,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    start_id=current_id
                )
                all_chunks.extend(web_chunks)

            # Build index if we have content
            if all_chunks:
                self.embeddings, self.chunks = self.embedding_handler.build_vector_index(all_chunks)
                self.embedding_handler.save_index_to_cache(self.embeddings, self.chunks)
                print(f"✅ Index built: {len(self.chunks)} chunks")
            else:
                print("⚠️ No content found")
        else:
            print(f"✅ Loaded from cache: {len(self.chunks)} chunks")

    def clear_cache(self) -> None:
        """Clear the cached embeddings and chunks."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print("✅ Cache cleared")
            # Reset in-memory data
            self.embeddings = np.array([])
            self.chunks = []
        else:
            print("⚠️ No cache to clear")

    def retrieve(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant vaccine information from the knowledge base.

        This function searches through official vaccine documents and web pages
        to find the most relevant information for answering the user's question.
        It uses semantic search with embeddings to find contextually similar content.

        The function is designed to be used as a tool by AI agents. It returns
        formatted text with source citations that the agent can use to formulate
        accurate, grounded responses.

        Args:
            query (str): The user's question about vaccines. This should be a natural
                language question such as "What vaccines are recommended for pregnant
                women?" or "What are the side effects of the flu vaccine?"

            k (int, optional): The number of most relevant text chunks to retrieve
                from the knowledge base. More chunks provide more context but may
                include less relevant information. Defaults to 5.
                Range: 1-10 recommended.

        Returns:
            str: A formatted string containing the retrieved information with source
                citations. Each piece of information is prefixed with [SOURCE: ...]
                indicating where it came from (either a PDF filename or a web URL).
                Multiple results are separated by "---" dividers.

                Returns an error message if:
                - The knowledge base is not initialized
                - No relevant information is found
                - An exception occurs during retrieval

        Examples:
            >>> result = rag_kb.retrieve("Are vaccines safe during pregnancy?")
            >>> print(result)
            [SOURCE: https://example.com/vaccines/pregnancy]
            Vaccines recommended during pregnancy include...

            ---

            [SOURCE: vaccine_guidelines.pdf]
            The flu vaccine is safe and recommended for all pregnant women...

            >>> result = rag_kb.retrieve("COVID-19 vaccine schedule", k=3)
            >>> # Returns top 3 most relevant chunks about COVID-19 vaccination

        Note:
            This function performs a similarity search using embeddings, so it may
            return relevant information even if the exact keywords don't match.
            Always check the sources to verify the information is appropriate for
            the specific question asked.
        """
        if not self.chunks:
            return "Error: Knowledge base not initialized. Build the knowledge base first."

        try:
            top_chunks = self.embedding_handler.retrieve_top_k(
                query, self.embeddings, self.chunks, k=k
            )
            if not top_chunks:
                return "No relevant information found."

            # Format results with source citations
            results = []
            for c in top_chunks:
                # Handle both old and new chunk formats
                if hasattr(c, 'doc_type'):
                    source = Path(str(c.source)).name if c.doc_type.value == "pdf" else str(c.source)
                else:
                    # Fallback for old cached chunks without doc_type
                    source = str(c.source)
                    if Path(source).suffix.lower() == '.pdf':
                        source = Path(source).name

                results.append(f"[SOURCE: {source}]\n{c.content}")

            return "\n\n---\n\n".join(results)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_stats(self) -> dict:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with knowledge base statistics
        """
        return {
            "num_chunks": len(self.chunks),
            "embedding_shape": self.embeddings.shape if self.embeddings.size > 0 else (0, 0),
            "cache_dir": str(self.cache_dir),
            "has_data": len(self.chunks) > 0
        }
