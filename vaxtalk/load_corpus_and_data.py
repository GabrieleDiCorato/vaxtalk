"""Load Corpus and Data: Clear cache and reload the entire corpus.

This script deletes the cache and rebuilds the knowledge base from scratch,
loading all PDFs and crawling the website again.

Usage:
    uv run load-corpus
"""

from pathlib import Path

# Project Imports
from vaxtalk.config import load_env_variables, get_env_variable, get_env_int
from vaxtalk.config.logging_config import setup_logging, get_logger
from vaxtalk.rag.rag_service import RagService


def main():
    """Main entry point for loading corpus and data.

    This function:
    1. Clears the cache directory
    2. Reloads all documents from PDFs
    3. Crawls the website
    4. Builds new embeddings
    5. Saves the new cache
    """
    # Get project root relative to this file's location
    project_root = Path(__file__).resolve().parent.parent

    # Setup logging
    logger = setup_logging(log_dir=project_root / "logs", log_level="INFO")
    logger.info("Starting corpus reload process...")
    logger.info("Project root: %s", project_root)

    # Load environment variables
    load_env_variables(project_root / ".env")
    GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY")

    # Configuration
    DOC_FOLDER_PATH = project_root / get_env_variable("DOC_FOLDER_PATH", "docs")
    CACHE_DIR = project_root / get_env_variable("CACHE_DIR", "cache")
    DOC_WEB_URL_ROOT = get_env_variable("DOC_WEB_URL_ROOT", "https://www.serviziterritoriali-asstmilano.it/servizi/vaccinazioni/")

    RAG_MAX_PAGES = get_env_int("RAG_MAX_PAGES", 10)
    RAG_MAX_DEPTH = get_env_int("RAG_MAX_DEPTH", 2)
    RAG_CHUNK_SIZE = get_env_int("RAG_CHUNK_SIZE", 800)
    RAG_CHUNK_OVERLAP = get_env_int("RAG_CHUNK_OVERLAP", 200)

    logger.info("Document folder: %s", DOC_FOLDER_PATH)
    logger.info("Cache directory: %s", CACHE_DIR)
    logger.info("Web URL root: %s", DOC_WEB_URL_ROOT)

    # Initialize knowledge base
    rag_kb = RagService(
        api_key=GOOGLE_API_KEY,
        cache_dir=CACHE_DIR
    )
    logger.info("Knowledge base initialized")

    # Clear existing cache
    logger.info("Clearing cache...")
    rag_kb.clear_cache()
    logger.info("Cache cleared")

    # Rebuild knowledge base
    logger.info("Building knowledge base from scratch...")
    rag_kb.build_knowledge_base(
        pdf_folder=DOC_FOLDER_PATH,
        root_url=DOC_WEB_URL_ROOT,
        max_pages=RAG_MAX_PAGES,
        max_depth=RAG_MAX_DEPTH,
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        use_cache=True,  # Will save to cache after building
    )

    # Display statistics
    stats = rag_kb.get_stats()
    logger.info("Knowledge Base Rebuilt Successfully!")
    logger.info("  Total chunks: %s", stats['num_chunks'])
    logger.info("  Embedding shape: %s", stats['embedding_shape'])
    logger.info("  Cache directory: %s", stats['cache_dir'])

    logger.info("Corpus reload complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
