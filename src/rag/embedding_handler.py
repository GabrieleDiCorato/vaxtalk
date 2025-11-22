"""
Embedding Handler for RAG System

This module provides functionality to create, cache, and retrieve embeddings
for use in a Retrieval-Augmented Generation (RAG) system.
"""

import pickle
import heapq
import numpy as np
from pathlib import Path
from google import genai
from src.model.document_chunk import DocumentChunk


class EmbeddingHandler:
    """
    Handles embedding generation, caching, and retrieval operations.

    This class provides methods to:
    - Generate embeddings using Google Gemini API
    - Cache embeddings and chunks to disk
    - Load embeddings from cache
    - Build vector indices from document chunks
    - Calculate cosine similarity between vectors
    - Retrieve top-k similar chunks for a query
    """

    def __init__(self, api_key: str, cache_dir: str | Path = "../cache"):
        """
        Initialize the EmbeddingHandler.

        Args:
            api_key: Google API key for Gemini
            cache_dir: Directory to store cached embeddings and chunks
        """
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = Path(cache_dir)
        self.embeddings_cache_file = self.cache_dir / "embeddings.pkl"
        self.chunks_cache_file = self.cache_dir / "chunks.pkl"

    def save_index_to_cache(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        """
        Save embeddings and chunks to disk cache for faster loading.

        Args:
            embeddings: Numpy array of embedding vectors
            chunks: List of DocumentChunk objects
        """
        self.cache_dir.mkdir(exist_ok=True)

        with open(self.embeddings_cache_file, 'wb') as f:
            pickle.dump(embeddings, f)

        with open(self.chunks_cache_file, 'wb') as f:
            pickle.dump(chunks, f)

        print(f"✅ Index cached to {self.cache_dir}")

    def load_index_from_cache(self) -> tuple[np.ndarray, list[DocumentChunk]]:
        """
        Load embeddings and chunks from disk cache.

        Returns:
            Tuple of (embeddings array, chunks list) or (empty array, empty list) if not cached
        """
        if not self.embeddings_cache_file.exists() or not self.chunks_cache_file.exists():
            return np.array([]), []

        try:
            with open(self.embeddings_cache_file, 'rb') as f:
                embeddings = pickle.load(f)

            with open(self.chunks_cache_file, 'rb') as f:
                chunks = pickle.load(f)

            print(f"✅ Index loaded from cache: {len(chunks)} chunks")
            return embeddings, chunks
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            return np.array([]), []

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate embeddings for a list of texts using Google Gemini API.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to embed per API call

        Returns:
            Numpy array of shape (n, d) where n is number of texts and d is embedding dimension
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        all_vectors = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"[EMB] Batch {i}–{i+len(batch)-1} of {len(texts)}")

                # Gemini embedding call
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=batch
                )
                # Extract embedding vectors from response
                vectors = [item.values for item in response.embeddings]
                all_vectors.extend(vectors)

            return np.array(all_vectors, dtype=np.float32)

        except Exception as e:
            print(f"❌ Error during embedding: {e}")
            raise

    def build_vector_index(self, chunks: list[DocumentChunk]) -> tuple[np.ndarray, list[DocumentChunk]]:
        """
        Create embedding vectors for all document chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Tuple of (embeddings array, chunks list)
        """
        print("[INDEX] Calculating embeddings for all chunks...")
        texts = [c.content for c in chunks]
        embeddings = self.embed_texts(texts)
        print(f"[INDEX] Embeddings shape: {embeddings.shape}")
        return embeddings, chunks

    @staticmethod
    def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between multiple vectors and a single query vector.

        Args:
            a: Matrix of shape (n, d) containing n vectors of dimension d
            b: Single vector of shape (d,)

        Returns:
            Array of shape (n,) with similarity scores
        """
        if a.size == 0:
            return np.array([])

        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        return np.dot(a_norm, b_norm)

    def retrieve_top_k(self,
                       query: str,
                       embeddings: np.ndarray,
                       chunks: list[DocumentChunk],
                       k: int = 5) -> list[DocumentChunk]:
        """
        Retrieve the top-k most similar chunks to a query.

        Embeds the query and finds the k chunks with highest cosine similarity.

        Args:
            query: Query string
            embeddings: Matrix of chunk embeddings (n, d)
            chunks: List of DocumentChunk objects corresponding to embeddings
            k: Number of top results to return

        Returns:
            List of top-k DocumentChunk objects, ordered by similarity (highest first)
        """
        if embeddings.size == 0 or not chunks:
            return []

        query_emb = self.embed_texts([query])[0]  # (d,)
        sims = self.cosine_similarity_matrix(embeddings, query_emb)  # (n,)

        # Get indices of top k similarities using heapq
        k = min(k, len(chunks))
        top_k_items = heapq.nlargest(k, enumerate(sims), key=lambda x: x[1])

        # Extract chunks in order of similarity (highest first)
        top_chunks = [chunks[idx] for idx, _ in top_k_items]
        return top_chunks
