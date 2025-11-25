"""
Sentiment Analysis Service

This module provides functionality to analyze emotional tone in text using a hybrid approach:
- LLM-based classification for nuanced understanding
- Embedding-based similarity matching against prototype emotions
- Fusion of both approaches for robust results

Supports three emotion dimensions: satisfaction, frustration, confusion
Each dimension is classified as: low, medium, high
"""

import os
import pickle
import json
import numpy as np
import requests
from pathlib import Path
from typing import Any
from google import genai
from google.genai.types import HttpRetryOptions
from google.adk.agents import Agent
from vaxtalk.config import get_env_variable, get_env_float
from vaxtalk.config.logging_config import get_logger
from vaxtalk.connectors.llm_connection_factory import LlmConnectionFactory
from vaxtalk.model.sentiment_output import SentimentOutput, Intensity

logger = get_logger(__name__)


class SentimentService:
    """
    Analyzes emotional tone of text using LLM + embedding fusion.

    Pre-loads emotion prototype embeddings at startup and caches them for efficient reuse.
    Combines LLM predictions with embedding similarity for robust sentiment analysis.
    """

    # Emotion categories and intensity levels
    EMOTIONS = ["satisfaction", "frustration", "confusion"]
    # We have to manually define a degree of severity for each level
    LEVELS = [Intensity.LOW, Intensity.MEDIUM, Intensity.HIGH]
    LEVEL_TO_NUM = {level: idx for idx, level in enumerate(LEVELS)}
    NUM_TO_LEVEL = {idx: level for idx, level in enumerate(LEVELS)}

    # LLM system prompt for sentiment classification
    LLM_SYSTEM_PROMPT = """
You are part of a multilingual medical information system focused on vaccines.

You analyze the emotional tone of a patient message, which may be written in any language.
Your task is to estimate the intensity (low, medium, high) of these three dimensions:

- satisfaction (reassured, positive, grateful)
- frustration (annoyed, angry, fed up)
- confusion (uncertain, lost, not understanding)

Constraints:
- Use only "low", "medium", or "high" as values.
- Classify all three emotions independently.
- Focus ONLY on the emotion in the text, not on medical correctness.
- If the message is very short or ambiguous, prefer "medium" for one emotion and "low" for the others.

Strict output format:
Return EXACTLY one JSON dictionary as a STRING, no extra text:

{
  "satisfaction": "<low|medium|high>",
  "frustration": "<low|medium|high>",
  "confusion": "<low|medium|high>"
}
"""

    def __init__(self):
        """
        Initialize the SentimentService.

        All configuration can be provided via environment variables.
        """
        # Determine project root
        self.project_root = Path(Path.cwd().parent.parent)

        # Read cache directory from environment or use default
        cache_dir = get_env_variable("CACHE_DIR", "cache")
        self.cache_dir = self.project_root / Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Read model configurations from environment
        self.embedding_model = get_env_variable("EMBEDDING_MODEL")
        self.llm_model = get_env_variable("SENTIMENT_LLM_MODEL")

        # We will use OpenRouter for everything
        self.openrouter_api_key = get_env_variable("OPENROUTER_API_KEY")

        # Read phrases file
        sentiment_phrases = get_env_variable(
                "SENTIMENT_PHRASES_FILE",
                "vaxtalk/sentiment/sentiment_phrases.json"
        )
        self.sentiment_phrases_file = Path(self.project_root / sentiment_phrases)

        self.sentiment_cache_file = self.cache_dir / "sentiment_prototypes.pkl"

        # Read fusion weights from environment or use defaults
        self.w_llm = get_env_float("SENTIMENT_W_LLM", 0.7)
        self.w_emb = get_env_float("SENTIMENT_W_EMB", 0.3)

        # Prototype storage
        self.sentiment_phrases: list[dict[str, Any]] | None = None

        logger.info("SentimentService initialized:")
        logger.info("  - Embedding model: %s", self.embedding_model)
        logger.info("  - LLM model: %s", self.llm_model)
        logger.info("  - Sentiment phrases file: %s", self.sentiment_phrases_file)
        logger.info("  - Fusion weights: LLM=%.2f, EMB=%.2f", self.w_llm, self.w_emb)

    def build_sentiment_phrases_embeddings(self, use_cache: bool = True) -> None:
        """
        Load sentiment phrases and cache normalized embeddings.

        If use_cache is True and cache exists, loads from cache.
        Otherwise, evaluate directly the embeddings at every startup.

        Args:
            use_cache: Whether to use cached prototypes if available
        """
        # Try to load from cache
        if use_cache and self.sentiment_cache_file.exists():
            try:
                with open(self.sentiment_cache_file, 'rb') as f:
                    self.sentiment_phrases = pickle.load(f)
                if self.sentiment_phrases:
                    logger.info("Loaded %s emotion prototypes from cache", len(self.sentiment_phrases))
                return
            except Exception as e:
                logger.warning("Error loading cached prototypes: %s. Rebuilding from JSON.", e)

        # Load from JSON file
        logger.info("Loading sentiment phrases from %s", self.sentiment_phrases_file)

        if not self.sentiment_phrases_file.exists():
            raise FileNotFoundError(f"Emotion prototypes file not found: {self.sentiment_phrases_file}")

        with open(self.sentiment_phrases_file, "r", encoding="utf-8") as f:
            sentiment_phrases = json.load(f)

        logger.info("Loaded %s prototypes from JSON file", len(sentiment_phrases))

        # Generate embeddings for each phrase
        for p in sentiment_phrases:
            text = p.get("text", "")
            emb = self._embed_single_text(text)
            p["embedding"] = emb.tolist()

            # Normalize embeddings for cosine similarity
            vec = np.asarray(p["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            p["_embedding_norm"] = vec

        self.sentiment_phrases = sentiment_phrases

        # Cache for future use
        with open(self.sentiment_cache_file, 'wb') as f:
            pickle.dump(self.sentiment_phrases, f)

        if self.sentiment_phrases:
            logger.info("Cached %s prototypes to %s", len(self.sentiment_phrases), self.sentiment_cache_file)

    def _embed_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using configured embedding model.
        Supports both Google Gemini and OpenRouter embedding models.

        Args:
            text: Text to embed

        Returns:
            L2-normalized embedding vector
        """
        try:

            # Use OpenRouter embeddings API
            headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
            }
            payload = {
                "model": self.embedding_model,
                "input": [text],
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                json=payload,
                headers=headers,
                timeout=60
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter embedding error {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            emb = np.asarray(data["data"][0]["embedding"], dtype=np.float32)

            # Normalize embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            return emb

        except Exception as e:
            logger.error("Error generating embedding: %s", e)
            raise

    @staticmethod
    def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(u, v))

    def _max_similarity_per_emotion(
        self, query_vec_norm: np.ndarray
    ) -> dict[str, float]:
        """
        Compute maximum cosine similarity to prototypes for each emotion.

        Args:
            query_vec_norm: Normalized query embedding vector

        Returns:
            Dict mapping emotion name to maximum similarity score
        """
        if self.sentiment_phrases is None:
            raise RuntimeError("Prototypes not loaded. Call build_prototype_embeddings() first.")

        # Index prototypes by emotion
        emo_to_vecs: dict[str, list[np.ndarray]] = {emo: [] for emo in self.EMOTIONS}
        for p in self.sentiment_phrases:
            emo = p.get("emotion")
            if emo in emo_to_vecs:
                emo_to_vecs[emo].append(p["_embedding_norm"])

        # Compute max similarity per emotion
        scores: dict[str, float] = {}
        for emo in self.EMOTIONS:
            vecs = emo_to_vecs[emo]
            if not vecs:
                scores[emo] = 0.0
                continue
            sims = [self._cosine_similarity(query_vec_norm, v) for v in vecs]
            scores[emo] = max(sims) if sims else 0.0

        return scores

    def _call_llm_for_sentiment(self, query: str) -> dict[str, Intensity]:
        """
        Call LLM to classify sentiment using configured LLM model.
        Supports both Google Gemini and OpenRouter LLM models.

        Args:
            query: User text to analyze

        Returns:
            Dict mapping emotion name to Intensity level
        """
        try:
            # Construct full prompt
            full_prompt = f"{self.LLM_SYSTEM_PROMPT}\n\nUser message:\n{query}"

            # Use OpenRouter via requests
            headers = {
                "Authorization": f"Bearer {self.llm_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.0,
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                raw_text = str(data["choices"][0]["message"]["content"])
            else:
                raise ValueError(f"Unsupported model type: {self.llm_model}")

            # Parse JSON from response
            pred = self._parse_emotion_dict(raw_text if raw_text else "")

            logger.debug("LLM sentiment prediction: %s", pred)
            return pred

        except Exception as e:
            logger.error("Error calling LLM for sentiment: %s", e)
            # Fallback to neutral sentiment
            return {emo: Intensity.LOW for emo in self.EMOTIONS}

    def _parse_emotion_dict(self, raw_text: str) -> dict[str, Intensity]:
        """
        Parse LLM output to extract emotion dictionary.

        Args:
            raw_text: Raw LLM response text

        Returns:
            Dict mapping emotion name to Intensity, with fallback to LOW for invalid/missing values
        """
        try:
            # Find JSON substring
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            json_str = raw_text[start:end]
        except ValueError:
            json_str = raw_text.strip()

        try:
            obj = json.loads(json_str)
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}

        # Extract and validate levels
        pred = {}
        for emo in self.EMOTIONS:
            # Support both old format ("satisfaction") and new format ("satisfaction_level")
            val = obj.get(f"{emo}_level") or obj.get(emo)
            if isinstance(val, str):
                norm = val.strip().lower()
                if norm == "low":
                    pred[emo] = Intensity.LOW
                elif norm == "medium":
                    pred[emo] = Intensity.MEDIUM
                elif norm == "high":
                    pred[emo] = Intensity.HIGH
                else:
                    pred[emo] = Intensity.LOW
            else:
                pred[emo] = Intensity.LOW

        return pred

    def analyze_emotion(self, query: str) -> SentimentOutput:
        """
        Analyze emotional tone of text using LLM + embedding fusion.

        Args:
            query: Text to analyze

        Returns:
            SentimentOutput with frustration_level, confusion_level, and satisfaction_level
        """
        if self.sentiment_phrases is None:
            raise RuntimeError("Prototypes not loaded. Call build_prototype_embeddings() first.")

        logger.debug("Analyzing emotion for query: %s", query[:100])

        # 1. Get LLM prediction
        llm_pred = self._call_llm_for_sentiment(query)

        # 2. Get embedding scores
        query_vec_norm = self._embed_single_text(query)
        emb_scores = self._max_similarity_per_emotion(query_vec_norm)

        # 3. Normalize embedding scores across emotions [0, 1] then scale to [0, 2]
        valid_scores = [s for s in emb_scores.values() if isinstance(s, (float, int))]
        if valid_scores:
            s_min = float(min(valid_scores))
            s_max = float(max(valid_scores))
            denom = (s_max - s_min) if (s_max > s_min) else 1.0
            emb_scaled = {}
            for emo in self.EMOTIONS:
                s = emb_scores.get(emo, 0.0)
                s_norm = (float(s) - s_min) / denom  # [0, 1]
                emb_scaled[emo] = 2.0 * s_norm  # [0, 2]
        else:
            emb_scaled = {emo: 0.0 for emo in self.EMOTIONS}

        # 4. Fusion per emotion
        fused_pred: dict[str, Intensity] = {}

        for emo in self.EMOTIONS:
            # LLM numeric
            llm_level = llm_pred.get(emo, Intensity.LOW)
            llm_num = self.LEVEL_TO_NUM.get(llm_level, 0)

            # Embedding numeric
            emb_num = emb_scaled.get(emo, 0.0)

            # Weighted fusion
            fused_raw = self.w_llm * llm_num + self.w_emb * emb_num

            # Round and clamp to [0, 2]
            fused_num = int(round(fused_raw))
            fused_num = max(0, min(2, fused_num))

            fused_pred[emo] = self.NUM_TO_LEVEL[fused_num]

        # Create SentimentOutput model
        result = SentimentOutput(
            satisfaction=fused_pred["satisfaction"],
            frustration=fused_pred["frustration"],
            confusion=fused_pred["confusion"]
        )

        logger.info("Sentiment analysis result: %s", result)
        return result

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about loaded prototypes.

        Returns:
            Dict with prototype counts by emotion
        """
        if self.sentiment_phrases is None:
            return {"total": 0, "by_emotion": {}}

        counts = {emo: 0 for emo in self.EMOTIONS}
        for p in self.sentiment_phrases:
            emo = p.get("emotion")
            if emo in counts:
                counts[emo] += 1

        return {
            "total": len(self.sentiment_phrases),
            "by_emotion": counts
        }
