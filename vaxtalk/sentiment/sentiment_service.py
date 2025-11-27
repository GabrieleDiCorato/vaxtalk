"""
Sentiment Analysis Service

This module provides functionality to analyze emotional tone in text using a hybrid approach:
- LLM-based classification for nuanced understanding
- Embedding-based similarity matching against prototype emotions
- Fusion of both approaches for robust results

Supports three emotion dimensions: satisfaction, frustration, confusion
Each dimension is classified as: low, medium, high
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import requests

from vaxtalk.config import get_env_float, get_env_variable
from vaxtalk.config.logging_config import get_logger
from vaxtalk.model.sentiment_output import Intensity, SentimentOutput
from vaxtalk.prompts import SENTIMENT_LLM_SYSTEM_PROMPT

logger = get_logger(__name__)


class SentimentService:
    """
    Analyzes emotional tone of text using LLM + embedding fusion.

    Pre-loads emotion prototype embeddings at startup and caches them for efficient reuse.
    Combines LLM predictions with embedding similarity for robust sentiment analysis.

    The service uses a hybrid approach:
    - LLM analysis for contextual understanding
    - Embedding similarity for pattern matching
    - Weighted fusion of both methods for final classification
    """

    # Emotion categories supported by the service
    EMOTIONS: list[str] = ["satisfaction", "frustration", "confusion"]

    # Intensity levels in ascending order of severity
    LEVELS: list[Intensity] = [Intensity.LOW, Intensity.MEDIUM, Intensity.HIGH]
    LEVEL_TO_NUM: dict[Intensity, int] = {level: idx for idx, level in enumerate(LEVELS)}
    NUM_TO_LEVEL: dict[int, Intensity] = {idx: level for idx, level in enumerate(LEVELS)}

    # Similarity score scaling range for normalization
    SIMILARITY_SCALE_MIN: float = 0.0
    SIMILARITY_SCALE_MAX: float = 2.0

    # API configuration constants
    API_TIMEOUT_SECONDS: int = 60
    MAX_LOG_QUERY_LENGTH: int = 100
    MAX_ERROR_MESSAGE_LENGTH: int = 200

    def __init__(self):
        """
        Initialize the SentimentService.

        All configuration is loaded from environment variables.
        Creates necessary directories and initializes internal state.
        """
        # Initialize internal state variables
        self.sentiment_phrases: list[dict[str, Any]] | None = None
        self._emotion_embeddings: dict[str, list[np.ndarray]] | None = None
        self._session: requests.Session | None = None

        self._load_configuration()
        self._log_initialization()

    def _load_configuration(self) -> None:
        """Load all configuration from environment variables and set up paths."""
        # Project structure
        self.project_root = Path(__file__).resolve().parent.parent.parent
        logger.info("Sentiment project root directory set to %s", self.project_root)

        # Cache directory setup
        cache_dir = get_env_variable("CACHE_DIR", "cache")
        self.cache_dir = self.project_root / Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Set sentiment cache directory to %s", self.cache_dir)

        # Model configurations (support legacy variable names for backward compatibility)
        self.embedding_model = get_env_variable("MODEL_SENTIMENT_EMBEDDING", "text-embedding-004")
        self.llm_model = get_env_variable("MODEL_SENTIMENT_LLM", "openrouter/mistralai/ministral-8b")

        # OpenRouter API configuration
        self.openrouter_api_key = get_env_variable("OPENROUTER_API_KEY")
        self.openrouter_embed_url = get_env_variable("OPENROUTER_EMBED_URL")
        self.openrouter_chat_url = get_env_variable("OPENROUTER_CHAT_URL")

        # Sentiment phrases file path
        sentiment_phrases_path = get_env_variable(
            "SENTIMENT_PHRASES_FILE",
            "sentiment/sentiment_phrases.json"
        )
        self.sentiment_phrases_file = self.project_root / sentiment_phrases_path
        self.sentiment_cache_file = self.cache_dir / "sentiment_prototypes.pkl"

        # Fusion weights for combining LLM and embedding predictions
        self.w_llm = get_env_float("SENTIMENT_W_LLM", 0.7)
        self.w_emb = get_env_float("SENTIMENT_W_EMB", 0.3)

    def _log_initialization(self) -> None:
        """Log initialization information for debugging."""
        logger.info("SentimentService initialized:")
        logger.info("  - Embedding model: %s", self.embedding_model)
        logger.info("  - LLM model: %s", self.llm_model)
        logger.info("  - Sentiment phrases file: %s", self.sentiment_phrases_file)
        logger.info("  - Fusion weights: LLM=%.2f, EMB=%.2f", self.w_llm, self.w_emb)

    @property
    def session(self) -> requests.Session:
        """
        Get or create a reusable HTTP session.

        Lazily initializes the session on first access. Using a session
        enables connection pooling and reuse across multiple API calls.

        Returns:
            Configured requests Session instance
        """
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def close(self) -> None:
        """
        Close the HTTP session and release resources.

        Should be called when the service is no longer needed to properly
        release connection pool resources.
        """
        if self._session is not None:
            self._session.close()
            self._session = None
            logger.debug("HTTP session closed")

    def _load_sentiment_phrases_from_file(self) -> list[dict[str, Any]]:
        """
        Load sentiment phrases from JSON file.

        Returns:
            List of sentiment phrase dictionaries with emotion labels and text

        Raises:
            FileNotFoundError: If the sentiment phrases file does not exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        logger.info("Loading sentiment phrases from %s", self.sentiment_phrases_file)

        if not self.sentiment_phrases_file.exists():
            raise FileNotFoundError(
                f"Sentiment phrases file not found: {self.sentiment_phrases_file}"
            )

        try:
            with open(self.sentiment_phrases_file, "r", encoding="utf-8") as f:
                sentiment_phrases: list[dict[str, Any]] = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in sentiment phrases file: %s", e)
            raise

        logger.info("Loaded %s sentiment phrases from JSON file", len(sentiment_phrases))
        return sentiment_phrases

    def build_sentiment_phrases_embeddings(self, use_cache: bool = True) -> None:
        """
        Load sentiment phrases and generate normalized embeddings.

        Attempts to load from cache if available and use_cache is True.
        Otherwise, generates embeddings from the JSON phrases file.

        Args:
            use_cache: Whether to use cached prototypes if available
        """
        if use_cache and self._try_load_from_cache():
            return

        self._build_and_cache_embeddings()

    def _try_load_from_cache(self) -> bool:
        """
        Attempt to load sentiment phrases from cache.

        Returns:
            True if successfully loaded from cache, False otherwise
        """
        if not self.sentiment_cache_file.exists():
            return False

        try:
            with open(self.sentiment_cache_file, 'rb') as f:
                self.sentiment_phrases = pickle.load(f)

            if self.sentiment_phrases:
                logger.info(
                    "Loaded %s emotion embeddings from cache",
                    len(self.sentiment_phrases)
                )
                self._build_emotion_embeddings_index()
                return True
            return False
        except Exception as e:
            logger.warning(
                "Error loading cached prototypes: %s. Rebuilding from JSON.",
                e
            )
            return False

    def _build_and_cache_embeddings(self) -> None:
        """
        Build embeddings from phrases file and cache the results.

        Loads phrases from JSON, generates embeddings, and saves to cache.
        """
        # Load sentiment expressions from JSON file
        sentiment_phrases = self._load_sentiment_phrases_from_file()

        # Generate embeddings for all phrases
        sentiment_phrases_with_embeddings = self._add_embeddings_to_phrases(
            sentiment_phrases
        )

        self.sentiment_phrases = sentiment_phrases_with_embeddings
        self._build_emotion_embeddings_index()
        self._save_to_cache()

    def _save_to_cache(self) -> None:
        """
        Save sentiment phrases with embeddings to cache file.
        """
        if not self.sentiment_phrases:
            return

        try:
            with open(self.sentiment_cache_file, 'wb') as f:
                pickle.dump(self.sentiment_phrases, f)
            logger.info(
                "Cached %s prototypes to %s",
                len(self.sentiment_phrases),
                self.sentiment_cache_file
            )
        except Exception as e:
            logger.error("Error saving to cache: %s", e)

    def clear_cache(self) -> None:
        """
        Clear cached sentiment prototype embeddings.

        Removes the cache file if it exists. Safe to call even if cache doesn't exist.
        """
        if not self.sentiment_cache_file.exists():
            logger.debug("Cache file does not exist, nothing to clear")
            return

        try:
            os.remove(self.sentiment_cache_file)
            logger.info(
                "Cleared sentiment prototype cache: %s",
                self.sentiment_cache_file
            )
        except OSError as e:
            logger.error("Error clearing sentiment cache: %s", e)
            raise

    def _build_emotion_embeddings_index(self) -> None:
        """
        Pre-index embeddings by emotion for efficient similarity computation.

        Creates a lookup dictionary mapping each emotion to its prototype embeddings.
        This enables O(1) access during similarity computation.
        """
        if self.sentiment_phrases is None:
            self._emotion_embeddings = None
            return

        # Initialize index with empty lists for each emotion
        emotion_index: dict[str, list[np.ndarray]] = {
            emotion: [] for emotion in self.EMOTIONS
        }

        # Populate index with embeddings from phrases
        for phrase in self.sentiment_phrases:
            emotion = phrase.get("emotion")
            embedding = phrase.get("embedding")

            if emotion in emotion_index and embedding is not None:
                emotion_index[emotion].append(embedding)

        self._emotion_embeddings = emotion_index

        total_embeddings = sum(len(vecs) for vecs in emotion_index.values())
        counts_by_emotion = {emo: len(vecs) for emo, vecs in emotion_index.items()}
        logger.debug(
            "Indexed %s emotion embeddings: %s",
            total_embeddings,
            counts_by_emotion
        )

    def _generate_normalized_embeddings(
        self,
        texts: list[str]
    ) -> list[np.ndarray]:
        """
        Generate L2-normalized embeddings for a list of texts using OpenRouter API.

        Args:
            texts: List of texts to embed

        Returns:
            List of L2-normalized embedding vectors in the same order as input texts

        Raises:
            RuntimeError: If the API request fails
            requests.RequestException: If there are network issues
        """
        if not texts:
            return []

        try:
            response = self._call_embedding_api(texts)
            embeddings = self._extract_and_normalize_embeddings(response)
            return embeddings
        except Exception as e:
            logger.error("Error generating batch embeddings: %s", e)
            raise

    def _add_embeddings_to_phrases(
        self,
        phrases: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Generate and attach embeddings to sentiment phrases.

        Args:
            phrases: List of phrase dictionaries without embeddings

        Returns:
            List of phrase dictionaries with embeddings attached
        """
        texts = [p["text"] for p in phrases]
        embeddings = self._generate_normalized_embeddings(texts)

        # Attach embeddings to phrases
        for phrase, embedding in zip(phrases, embeddings):
            phrase["embedding"] = embedding

        return phrases

    def _call_embedding_api(self, texts: list[str]) -> dict[str, Any]:
        """
        Call OpenRouter embedding API.

        Args:
            texts: List of texts to embed

        Returns:
            API response as dictionary

        Raises:
            RuntimeError: If the API returns an error status
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.embedding_model,
            "input": texts,
        }

        response = self.session.post(
            self.openrouter_embed_url,
            json=payload,
            headers=headers,
            timeout=self.API_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            error_message = (
                f"OpenRouter embedding error {response.status_code}: "
                f"{response.text[:self.MAX_ERROR_MESSAGE_LENGTH]}"
            )
            raise RuntimeError(error_message)

        return response.json()

    def _extract_and_normalize_embeddings(
        self,
        api_response: dict[str, Any]
    ) -> list[np.ndarray]:
        """
        Extract embeddings from API response and normalize them.

        Args:
            api_response: Response dictionary from embedding API

        Returns:
            List of L2-normalized embedding vectors
        """
        embeddings: list[np.ndarray] = []

        for item in api_response.get("data", []):
            raw_embedding = np.asarray(item["embedding"], dtype=np.float32)
            normalized_embedding = self._normalize_vector(raw_embedding)
            embeddings.append(normalized_embedding)

        return embeddings

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length (L2 normalization).

        Args:
            vector: Input vector to normalize

        Returns:
            L2-normalized vector (or original if norm is zero)
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    @staticmethod
    def _compute_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate cosine similarity between two normalized vectors.

        Since vectors are pre-normalized, cosine similarity reduces to dot product.

        Args:
            u: First normalized vector
            v: Second normalized vector

        Returns:
            Cosine similarity score in range [-1, 1]
        """
        return float(np.dot(u, v))

    def _compute_max_similarity_per_emotion(
        self,
        query_embedding: np.ndarray
    ) -> dict[str, float]:
        """
        Compute maximum cosine similarity to prototypes for each emotion.

        For each emotion category, finds the prototype with highest similarity
        to the query embedding.

        Args:
            query_embedding: Normalized query embedding vector

        Returns:
            Dictionary mapping emotion name to maximum similarity score

        Raises:
            RuntimeError: If embeddings haven't been loaded yet
        """
        if self._emotion_embeddings is None:
            raise RuntimeError(
                "Sentiment embeddings not loaded. "
                "Call build_sentiment_phrases_embeddings() first."
            )

        similarity_scores: dict[str, float] = {}

        for emotion in self.EMOTIONS:
            prototype_embeddings = self._emotion_embeddings[emotion]

            if not prototype_embeddings:
                similarity_scores[emotion] = 0.0
                continue

            similarities = [
                self._compute_cosine_similarity(query_embedding, prototype)
                for prototype in prototype_embeddings
            ]
            similarity_scores[emotion] = max(similarities)

        return similarity_scores

    def _get_llm_sentiment_prediction(self, query: str) -> dict[str, Intensity]:
        """
        Get sentiment classification from LLM using OpenRouter API.

        Args:
            query: User text to analyze

        Returns:
            Dictionary mapping emotion name to Intensity level
        """
        try:
            response_text = self._call_llm_api(query)
            prediction = self._parse_llm_response(response_text)
            logger.debug("LLM sentiment prediction: %s", prediction)
            return prediction
        except Exception as e:
            logger.error("Error calling LLM for sentiment: %s", e)
            return self._get_neutral_sentiment()

    def _call_llm_api(self, query: str) -> str:
        """
        Call OpenRouter LLM API for sentiment classification.

        Args:
            query: User text to analyze

        Returns:
            Raw text response from LLM

        Raises:
            RuntimeError: If API request fails
        """
        full_prompt = f"{SENTIMENT_LLM_SYSTEM_PROMPT}\n\nUser message:\n{query}"

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.0,
        }

        response = self.session.post(
            self.openrouter_chat_url,
            headers=headers,
            json=payload,
            timeout=self.API_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            error_message = (
                f"OpenRouter error {response.status_code}: "
                f"{response.text[:self.MAX_ERROR_MESSAGE_LENGTH]}"
            )
            raise RuntimeError(error_message)

        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def _get_neutral_sentiment(self) -> dict[str, Intensity]:
        """
        Get neutral sentiment prediction as fallback.

        Returns:
            Dictionary with all emotions set to LOW intensity
        """
        return {emotion: Intensity.LOW for emotion in self.EMOTIONS}

    def _parse_llm_response(self, raw_text: str) -> dict[str, Intensity]:
        """
        Parse LLM output to extract emotion intensity dictionary.

        Attempts to extract JSON from the response and map emotion values to
        Intensity enums. Falls back to LOW intensity for invalid values.

        Args:
            raw_text: Raw LLM response text

        Returns:
            Dictionary mapping emotion name to Intensity level
        """
        json_dict = self._extract_json_from_text(raw_text)
        return self._map_emotions_to_intensities(json_dict)

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """
        Extract JSON object from text that may contain additional content.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed dictionary, or empty dict if parsing fails
        """
        try:
            # Try to find JSON substring between braces
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
        except ValueError:
            json_str = text.strip()

        try:
            parsed = json.loads(json_str)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return {}

    def _map_emotions_to_intensities(
        self,
        emotion_dict: dict[str, Any]
    ) -> dict[str, Intensity]:
        """
        Map emotion strings from LLM to Intensity enum values.

        Supports both formats: "emotion" and "emotion_level" keys.

        Args:
            emotion_dict: Dictionary with emotion names and string intensity values

        Returns:
            Dictionary mapping emotion names to Intensity enums
        """
        result: dict[str, Intensity] = {}

        for emotion in self.EMOTIONS:
            # Support both "satisfaction" and "satisfaction_level" keys
            value = emotion_dict.get(f"{emotion}_level") or emotion_dict.get(emotion)
            result[emotion] = self._parse_intensity_value(value)

        return result

    @staticmethod
    def _parse_intensity_value(value: Any) -> Intensity:
        """
        Parse a single intensity value to Intensity enum.

        Args:
            value: Value to parse (expected to be string)

        Returns:
            Intensity enum, defaults to LOW for invalid values
        """
        if not isinstance(value, str):
            return Intensity.LOW

        normalized = value.strip().lower()
        intensity_map = {
            "low": Intensity.LOW,
            "medium": Intensity.MEDIUM,
            "high": Intensity.HIGH,
        }

        return intensity_map.get(normalized, Intensity.LOW)

    def analyze_emotion(self, query: str) -> SentimentOutput:
        """
        Analyze emotional tone of text using LLM + embedding fusion.

        Combines predictions from both LLM analysis and embedding similarity
        using weighted fusion to produce final sentiment classification.

        Args:
            query: Text to analyze

        Returns:
            SentimentOutput with satisfaction, frustration, and confusion levels

        Raises:
            RuntimeError: If sentiment phrases haven't been loaded yet
        """
        self._validate_embeddings_loaded()
        logger.debug("Analyzing emotion for query: %s", query[:self.MAX_LOG_QUERY_LENGTH])

        # Get predictions from both sources
        llm_prediction = self._get_llm_sentiment_prediction(query)
        embedding_scores = self._get_embedding_similarity_scores(query)

        # Normalize and scale embedding scores to [0, 2] range
        scaled_scores = self._scale_embedding_scores(embedding_scores)

        # Fuse predictions from both sources
        fused_prediction = self._fuse_predictions(llm_prediction, scaled_scores)

        # Create output model
        result = SentimentOutput(
            satisfaction=fused_prediction["satisfaction"],
            frustration=fused_prediction["frustration"],
            confusion=fused_prediction["confusion"]
        )

        logger.info("Sentiment analysis result: %s", result)
        return result

    def _validate_embeddings_loaded(self) -> None:
        """
        Validate that sentiment phrases have been loaded.

        Raises:
            RuntimeError: If sentiment phrases haven't been loaded
        """
        if self.sentiment_phrases is None:
            raise RuntimeError(
                "Prototypes not loaded. "
                "Call build_sentiment_phrases_embeddings() first."
            )

    def _get_embedding_similarity_scores(self, query: str) -> dict[str, float]:
        """
        Get embedding similarity scores for the query.

        Args:
            query: Text to analyze

        Returns:
            Dictionary mapping emotions to similarity scores
        """
        query_embeddings = self._generate_normalized_embeddings([query])
        query_embedding = query_embeddings[0]
        return self._compute_max_similarity_per_emotion(query_embedding)

    def _scale_embedding_scores(
        self,
        scores: dict[str, float]
    ) -> dict[str, float]:
        """
        Normalize and scale embedding scores to [0, 2] range.

        Normalizes scores relative to min/max, then scales to match
        the intensity level range [0, 2].

        Args:
            scores: Raw similarity scores by emotion

        Returns:
            Scaled scores in [0, 2] range
        """
        valid_scores = [
            score for score in scores.values()
            if isinstance(score, (float, int))
        ]

        if not valid_scores:
            return {emotion: 0.0 for emotion in self.EMOTIONS}

        min_score = float(min(valid_scores))
        max_score = float(max(valid_scores))
        score_range = max_score - min_score if max_score > min_score else 1.0

        scaled_scores = {}
        for emotion in self.EMOTIONS:
            score = scores.get(emotion, 0.0)
            # Normalize to [0, 1] then scale to [0, 2]
            normalized = (float(score) - min_score) / score_range
            scaled_scores[emotion] = self.SIMILARITY_SCALE_MAX * normalized

        return scaled_scores

    def _fuse_predictions(
        self,
        llm_prediction: dict[str, Intensity],
        embedding_scores: dict[str, float]
    ) -> dict[str, Intensity]:
        """
        Fuse LLM and embedding predictions using weighted combination.

        Args:
            llm_prediction: Intensity predictions from LLM
            embedding_scores: Scaled similarity scores from embeddings

        Returns:
            Dictionary mapping emotions to fused Intensity levels
        """
        fused: dict[str, Intensity] = {}

        for emotion in self.EMOTIONS:
            # Convert LLM prediction to numeric
            llm_intensity = llm_prediction.get(emotion, Intensity.LOW)
            llm_numeric = self.LEVEL_TO_NUM.get(llm_intensity, 0)

            # Get embedding score
            embedding_numeric = embedding_scores.get(emotion, 0.0)

            # Weighted fusion
            fused_value = (
                self.w_llm * llm_numeric + self.w_emb * embedding_numeric
            )

            # Round and clamp to valid intensity range [0, 2]
            fused_numeric = int(round(fused_value))
            fused_numeric = max(0, min(2, fused_numeric))

            fused[emotion] = self.NUM_TO_LEVEL[fused_numeric]

        return fused

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about loaded sentiment prototypes.

        Returns:
            Dictionary containing:
                - total: Total number of prototypes
                - by_emotion: Breakdown of prototype counts per emotion
        """
        if self.sentiment_phrases is None:
            return {"total": 0, "by_emotion": {}}

        emotion_counts = self._count_phrases_by_emotion()

        return {
            "total": len(self.sentiment_phrases),
            "by_emotion": emotion_counts
        }

    def _count_phrases_by_emotion(self) -> dict[str, int]:
        """
        Count sentiment phrases by emotion category.

        Returns:
            Dictionary mapping emotion names to phrase counts
        """
        if self.sentiment_phrases is None:
            return {}

        counts = {emotion: 0 for emotion in self.EMOTIONS}

        for phrase in self.sentiment_phrases:
            emotion = phrase.get("emotion")
            if emotion in counts:
                counts[emotion] += 1

        return counts
