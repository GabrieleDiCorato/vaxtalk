"""
Web Handler for RAG System

This module provides functionality to crawl websites and extract text content
for use in a Retrieval-Augmented Generation (RAG) system.
"""

import time
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from src.model.document_chunk import DocumentChunk, DocType


class WebHandler:
    """
    Handles web scraping and text extraction from websites.

    This class provides methods to:
    - Check if URLs belong to the same domain
    - Extract clean text from HTML pages
    - Crawl websites following internal links
    - Chunk text from web pages
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
    def is_same_domain(url: str, root_netloc: str) -> bool:
        """Check if a URL belongs to the same domain."""
        try:
            return urlparse(url).netloc == root_netloc
        except Exception:
            return False

    @staticmethod
    def extract_text_from_html(html: str) -> str:
        """Extract clean text from HTML, removing scripts and styles."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        # Normalize whitespace
        return " ".join(text.split())

    @staticmethod
    def crawl_website(root_url: str,
                      max_pages: int = 100,
                      max_depth: int = 3,
                      start_id: int = 0,
                      chunk_size: int = 800,
                      chunk_overlap: int = 200) -> list[DocumentChunk]:
        """
        Crawl a website and extract text chunks from all pages.

        Uses BFS to follow internal links up to a maximum depth.
        Only crawls pages from the same domain as the root URL.

        Args:
            root_url: Starting URL for the crawl
            max_pages: Maximum number of pages to visit
            max_depth: Maximum link depth from root
            start_id: Starting ID for chunks
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks

        Returns:
            List of DocumentChunk objects from all crawled pages
        """
        parsed_root = urlparse(root_url)
        root_netloc = parsed_root.netloc

        visited = set()
        to_visit: list[tuple[str, int]] = [(root_url, 0)]
        all_chunks: list[DocumentChunk] = []
        current_id = start_id

        session = requests.Session()
        session.headers.update({"User-Agent": "rag-bot/1.0"})

        while to_visit and len(visited) < max_pages:
            url, depth = to_visit.pop(0)

            if url in visited:
                continue
            visited.add(url)

            if depth > max_depth:
                continue

            try:
                print(f"[WEB] Downloading ({depth}): {url}")
                resp = session.get(url, timeout=10)
                if "text/html" not in resp.headers.get("Content-Type", ""):
                    continue

                text = WebHandler.extract_text_from_html(resp.text)
                if text.strip():
                    chunks = WebHandler.chunk_text(
                        text=text,
                        source=url,
                        doc_type=DocType.WEB,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        start_id=current_id
                    )
                    all_chunks.extend(chunks)
                    current_id = all_chunks[-1].id + 1 if all_chunks else current_id

                # Extract links and add to queue
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = str(a["href"])
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)

                    # Keep only HTTP/HTTPS links from same domain
                    if parsed.scheme not in ("http", "https"):
                        continue
                    if not WebHandler.is_same_domain(full_url, root_netloc):
                        continue
                    if full_url not in visited:
                        to_visit.append((full_url, depth + 1))

                # Polite crawling delay
                time.sleep(0.2)

            except Exception as e:
                print(f"[WEB] ERROR on {url}: {e}")

        print(f"[WEB] Total pages visited: {len(visited)}")
        print(f"[WEB] Total chunks from site: {len(all_chunks)}")
        return all_chunks
