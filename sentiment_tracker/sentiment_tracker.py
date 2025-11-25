import os
import json
from typing import Dict, Any, List, Optional

import requests
import numpy as np

# If you're using a .env file, remember to:
from dotenv import load_dotenv
load_dotenv()

# ======================================================
# Global config & constants
# ======================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

# Emotions & levels
EMOTIONS = ["satisfaction", "frustration", "confusion"]
LEVELS = ["low", "medium", "high"]
LEVEL_TO_NUM = {"low": 0, "medium": 1, "high": 2}
NUM_TO_LEVEL = {v: k for k, v in LEVEL_TO_NUM.items()}

# Default embedding setup
DEFAULT_EMBEDDING_MODEL = "google/gemini-embedding-001"  # adjust if needed for OpenRouter
PROTOTYPE_EMB_FILE = "emotion_prototypes_embeddings.json"

# Simple global cache for prototypes (to avoid re-reading file every call)
_PROTOTYPES_CACHE: Optional[List[Dict[str, Any]]] = None

#Define LLM system prompt (can reuse your earlier one)
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
""".strip()

#Pick LLM model
LLM_MODEL = "mistralai/ministral-8b"  # or any other OpenRouter chat model


# ======================================================
# Helper: load prototypes from JSON and pre-normalize
# ======================================================

def load_prototypes(proto_file: str = PROTOTYPE_EMB_FILE) -> List[Dict[str, Any]]:
    """
    Load prototype embeddings from a JSON file of the form:
    [
      {
        "id": "...",
        "language": "...",
        "emotion": "satisfaction" | "frustration" | "confusion",
        "text": "...",
        "embedding": [float, float, ...]
      },
      ...
    ]

    Adds a key "_embedding_norm" with the L2-normalized numpy vector.
    Caches the result so multiple calls are cheap.
    """
    global _PROTOTYPES_CACHE
    if _PROTOTYPES_CACHE is not None:
        return _PROTOTYPES_CACHE

    if not os.path.exists(proto_file):
        raise FileNotFoundError(f"Prototype embedding file not found: {proto_file}")

    with open(proto_file, "r", encoding="utf-8") as f:
        prototypes = json.load(f)

    for p in prototypes:
        vec = np.asarray(p["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        p["_embedding_norm"] = vec

    _PROTOTYPES_CACHE = prototypes
    return prototypes


# ======================================================
# Helper: LLM call via OpenRouter
# ======================================================

def call_openrouter_llm(
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: int = 60,
) -> str:
    """
    Call OpenRouter Chat Completions API with (system, user) messages.
    Returns the assistant's text content.
    """
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                continue

            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = repr(e)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


# ======================================================
# Helper: parse LLM JSON dict (emotion -> level)
# ======================================================

def extract_emotion_dict(raw_text: str) -> Dict[str, str]:
    """
    Try to parse the LLM output as a JSON dictionary with the required keys.
    We are tolerant to extra text as long as there is a JSON-looking substring.

    Returns dict: emotion -> level ("low"/"medium"/"high"),
    falling back to "low" if missing or invalid.
    """
    try:
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

    pred = {}
    for emo in EMOTIONS:
        val = obj.get(emo)
        if isinstance(val, str):
            norm = val.strip().lower()
            if norm in LEVELS:
                pred[emo] = norm
            else:
                pred[emo] = "low"
        else:
            pred[emo] = "low"
    return pred


# ======================================================
# Helper: embedding of a single text
# ======================================================

def embed_single_text(text: str, embedding_model: str) -> np.ndarray:
    """
    Get embedding vector for a single text via OpenRouter embeddings endpoint.
    Returns L2-normalized numpy array.
    """
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": embedding_model,
        "input": [text],
    }
    resp = requests.post(OPENROUTER_EMBED_URL, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    emb = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))


# ======================================================
# Helper: max similarity per emotion
# ======================================================

def max_sim_per_emotion(
    query_vec_norm: np.ndarray,
    prototypes: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    For a normalized query embedding, compute the maximum cosine similarity
    to prototypes for each emotion.
    """
    # index prototypes by emotion
    emo_to_vecs: Dict[str, List[np.ndarray]] = {emo: [] for emo in EMOTIONS}
    for p in prototypes:
        emo = p.get("emotion")
        if emo in emo_to_vecs:
            emo_to_vecs[emo].append(p["_embedding_norm"])

    scores: Dict[str, float] = {}
    for emo in EMOTIONS:
        vecs = emo_to_vecs[emo]
        if not vecs:
            scores[emo] = None
            continue
        sims = [cosine_sim(query_vec_norm, v) for v in vecs]
        scores[emo] = max(sims) if sims else None
    return scores


# ======================================================
# Main function: query -> fused emotion dict
# ======================================================

def analyze_emotions_fused(
    query: str,
    llm_model: str,
    llm_system_prompt: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    proto_file: str = PROTOTYPE_EMB_FILE,
    w_llm: float = 0.7,
    w_emb: float = 0.3,
) -> Dict[str, str]:
    """
    Given a user query, returns a fused emotion dictionary:

    {
      "satisfaction": "low|medium|high",
      "frustration": "low|medium|high",
      "confusion": "low|medium|high"
    }

    - LLM part:
        * uses llm_model + llm_system_prompt via OpenRouter Chat.
        * expects the LLM to return a JSON dict with those 3 keys.
    - Embedding part:
        * loads prototypes from proto_file (emotion_prototypes_embeddings.json).
        * embeds the query with embedding_model.
        * computes max similarity per emotion.

    - Fusion:
        * map LLM levels to 0/1/2.
        * normalize embedding scores across emotions for this query to [0,1],
          then scale to [0,2].
        * fused_raw = w_llm * llm_num + w_emb * emb_num
        * round, clamp to 0..2, map back to low/medium/high.
    """

    # --- 1. LLM prediction ---
    raw_llm_output = call_openrouter_llm(
        model=llm_model,
        system_prompt=llm_system_prompt,
        user_text=query,
        temperature=0.0,
    )
    llm_pred = extract_emotion_dict(raw_llm_output)  # emotion -> low/medium/high

    # --- 2. Embedding scores ---
    prototypes = load_prototypes(proto_file)
    query_vec_norm = embed_single_text(query, embedding_model=embedding_model)
    emb_scores = max_sim_per_emotion(query_vec_norm, prototypes)  # emotion -> float

    # normalize embedding scores per query
    valid_scores = [s for s in emb_scores.values() if isinstance(s, (float, int))]
    if valid_scores:
        s_min = float(min(valid_scores))
        s_max = float(max(valid_scores))
        denom = (s_max - s_min) if (s_max > s_min) else 1.0
        emb_scaled = {}
        for emo in EMOTIONS:
            s = emb_scores.get(emo)
            if not isinstance(s, (float, int)):
                emb_scaled[emo] = 0.0
            else:
                s_norm = (float(s) - s_min) / denom  # [0,1]
                emb_scaled[emo] = 2.0 * s_norm       # [0,2]
    else:
        emb_scaled = {emo: 0.0 for emo in EMOTIONS}

    # --- 3. Fusion per emotion ---
    fused_pred: Dict[str, str] = {}

    for emo in EMOTIONS:
        # LLM numeric
        llm_level = llm_pred.get(emo, "low")
        llm_num = LEVEL_TO_NUM.get(llm_level, 0)

        # Embedding numeric
        emb_num = emb_scaled.get(emo, 0.0)

        # Weighted fusion
        fused_raw = w_llm * llm_num + w_emb * emb_num

        # Round & clamp to [0, 2]
        fused_num = int(round(fused_raw))
        fused_num = max(0, min(2, fused_num))

        fused_pred[emo] = NUM_TO_LEVEL[fused_num]

    return fused_pred
