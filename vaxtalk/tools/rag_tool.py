"""
RAG Retrieval Tool

Provides the retrieve_info FunctionTool for querying the vaccine knowledge base.
"""

from pathlib import Path

from google.adk.tools.function_tool import FunctionTool

from vaxtalk.config import load_env_variables, get_env_variable, get_env_int
from vaxtalk.config.logging_config import get_logger
from vaxtalk.rag.rag_service import RagService

logger = get_logger(__name__)

######################################
## CONFIGURATION
######################################

project_root = Path(__file__).resolve().parent.parent.parent

load_env_variables(project_root / ".env")

GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY")
CACHE_DIR = project_root / get_env_variable("CACHE_DIR", "cache")
DOC_FOLDER_PATH = project_root / get_env_variable("DOC_FOLDER_PATH", "docs")
DOC_WEB_URL_ROOT = get_env_variable("DOC_WEB_URL_ROOT", None)

# RAG Configuration
RAG_MAX_PAGES = get_env_int("RAG_MAX_PAGES", 100)
RAG_MAX_DEPTH = get_env_int("RAG_MAX_DEPTH", 5)
RAG_CHUNK_SIZE = get_env_int("RAG_CHUNK_SIZE", 800)
RAG_CHUNK_OVERLAP = get_env_int("RAG_CHUNK_OVERLAP", 200)
RAG_RETRIEVAL_K = get_env_int("RAG_RETRIEVAL_K", 5)

######################################
## RAG KNOWLEDGE BASE SETUP
######################################

rag_kb = RagService(
    api_key=GOOGLE_API_KEY,
    cache_dir=CACHE_DIR
)
logger.info("Knowledge base initialized")

# Build knowledge base from PDFs and website
rag_kb.build_knowledge_base(
    pdf_folder=DOC_FOLDER_PATH,
    root_url=DOC_WEB_URL_ROOT,
    max_pages=RAG_MAX_PAGES,
    max_depth=RAG_MAX_DEPTH,
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
    use_cache=True,
)

# Display statistics
stats = rag_kb.get_stats()
logger.info("Knowledge Base Stats:")
logger.info("  Chunks: %s", stats['num_chunks'])
logger.info("  Embedding shape: %s", stats['embedding_shape'])

######################################
## RAG TOOL
######################################


def retrieve_info(query: str) -> str:
    """
    Retrieve relevant vaccine information from the knowledge base.

    Args:
        query: The user's question about vaccines

    Returns:
        Formatted string with relevant information and sources

    Examples:
        >>> result = retrieve_info("What are the side effects of COVID vaccines?")
        >>> print(result)  # Returns relevant chunks with source citations
    """
    return rag_kb.retrieve(query, k=RAG_RETRIEVAL_K)


# Create retrieval tool from the knowledge base
rag_tool = FunctionTool(retrieve_info)

logger.info("RAG tool configured")
