"""
Sentiment Analysis Tool

Provides the run_sentiment_analysis FunctionTool for analyzing user emotional state
and triggering escalation notifications when thresholds are exceeded.
"""

import os
from pathlib import Path
from typing import Any

import requests
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Content

from vaxtalk.config import load_env_variables
from vaxtalk.config.logging_config import get_logger
from vaxtalk.model import SentimentOutput, Intensity
from vaxtalk.sentiment.sentiment_service import SentimentService

logger = get_logger(__name__)

######################################
## CONFIGURATION
######################################

project_root = Path(__file__).resolve().parent.parent.parent
load_env_variables(project_root / ".env")

# Escalation Configuration
ESCALATION_ENABLED = os.getenv("ESCALATION_ENABLED", "true").strip().lower() not in {"false", "0", "no"}
ESCALATION_DIMENSIONS = tuple(
    dim.strip().lower()
    for dim in os.getenv("ESCALATION_DIMENSIONS", "frustration,confusion").split(",")
    if dim.strip()
)
ESCALATION_TRIGGER_LEVEL = os.getenv("ESCALATION_TRIGGER_LEVEL", "high").strip().lower()
ESCALATION_NOTICE_TEXT = os.getenv(
    "ESCALATION_NOTICE_TEXT",
    "A human specialist has been notified and may join the conversation.",
).strip()
DEFAULT_ESCALATION_TEMPLATE = (
    "VaxTalk escalation triggered.\n"
    "Session ID: {session_id}\n"
    "User name: {user_name}\n"
    "User message: {user_input}\n"
    "Sentiment: {sentiment_summary}"
)
ESCALATION_MESSAGE_TEMPLATE = os.getenv(
    "ESCALATION_MESSAGE_TEMPLATE",
    DEFAULT_ESCALATION_TEMPLATE,
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

INTENSITY_RANK: dict[Intensity, int] = {
    Intensity.LOW: 0,
    Intensity.MEDIUM: 1,
    Intensity.HIGH: 2,
}

try:
    ESCALATION_TRIGGER_INTENSITY = Intensity(ESCALATION_TRIGGER_LEVEL)
except ValueError:
    logger.warning(
        "Invalid ESCALATION_TRIGGER_LEVEL '%s'; defaulting to HIGH.",
        ESCALATION_TRIGGER_LEVEL,
    )
    ESCALATION_TRIGGER_INTENSITY = Intensity.HIGH

######################################
## SENTIMENT SERVICE SETUP
######################################

try:
    sentiment_service: SentimentService | None = SentimentService()
    sentiment_service.build_sentiment_phrases_embeddings(use_cache=True)
    proto_stats = sentiment_service.get_stats()
    logger.info(
        "SentimentService initialized with %s prototypes",
        proto_stats.get("total", 0)
    )
except Exception as exc:
    logger.error("Failed to initialize SentimentService: %s", exc)
    sentiment_service = None

######################################
## HELPER FUNCTIONS
######################################


def _neutral_sentiment_output() -> SentimentOutput:
    """Return a neutral sentiment with all dimensions set to LOW."""
    return SentimentOutput(
        satisfaction=Intensity.LOW,
        frustration=Intensity.LOW,
        confusion=Intensity.LOW,
    )


def _extract_user_input(tool_context: ToolContext) -> str:
    """
    Best-effort extraction of the latest user utterance for tools.

    Args:
        tool_context: The ADK tool context containing user content

    Returns:
        Extracted user text or empty string if unavailable
    """
    content: Content | None = getattr(tool_context, "user_content", None)
    if not content or not content.parts:
        return ""

    segments: list[str] = []
    for part in content.parts:
        if part.text:
            segments.append(part.text)
        elif part.inline_data and part.inline_data.data:
            try:
                segments.append(part.inline_data.data.decode("utf-8"))
            except UnicodeDecodeError:
                continue

    return "\n".join(segment.strip() for segment in segments if segment.strip())


def _clean_optional_str(value: Any) -> str | None:
    """Clean and validate optional string values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_named_field(source: Any, candidates: tuple[str, ...]) -> str | None:
    """
    Extract a field value from source using candidate field names.

    Args:
        source: Object or dict to extract from
        candidates: Tuple of field names to try in order

    Returns:
        First non-empty value found or None
    """
    if source is None:
        return None
    if isinstance(source, dict):
        for key in candidates:
            result = _clean_optional_str(source.get(key))
            if result:
                return result
        return None
    for key in candidates:
        result = _clean_optional_str(getattr(source, key, None))
        if result:
            return result
    return None


def _extract_session_metadata(tool_context: ToolContext) -> dict[str, str]:
    """
    Extract session ID and user name from tool context.

    Tries multiple possible locations where ADK might store this information.

    Args:
        tool_context: The ADK tool context

    Returns:
        Dict with 'session_id' and 'user_name' keys (with fallback values)
    """
    # Try to extract session_id from various locations
    session_id = _clean_optional_str(getattr(tool_context, "session_id", None))
    if not session_id:
        session_state = getattr(tool_context, "session_state", None)
        session_id = _clean_optional_str(
            getattr(session_state, "session_id", None)
        ) or _clean_optional_str(getattr(session_state, "id", None))
    if not session_id:
        session = getattr(tool_context, "session", None)
        session_id = _clean_optional_str(
            getattr(session, "session_id", None)
        ) or _clean_optional_str(getattr(session, "id", None))
    if not session_id and isinstance(getattr(tool_context, "state", None), dict):
        session_id = _clean_optional_str(
            tool_context.state.get("session_id")
        ) or _clean_optional_str(tool_context.state.get("session:id"))

    # Try to extract user_name from various locations
    user_name = _clean_optional_str(getattr(tool_context, "user_name", None))
    if not user_name:
        user_name = _extract_named_field(
            getattr(tool_context, "user", None),
            ("display_name", "name", "user_name"),
        )
    if not user_name:
        user_name = _extract_named_field(
            getattr(tool_context, "user_metadata", None),
            ("display_name", "name", "user_name"),
        )
    if not user_name and isinstance(getattr(tool_context, "state", None), dict):
        profile = tool_context.state.get("user_profile")
        user_name = _extract_named_field(
            profile,
            ("display_name", "name", "user_name"),
        ) or _clean_optional_str(tool_context.state.get("user_name"))

    return {
        "session_id": session_id or "<unknown session>",
        "user_name": user_name or "<unknown user>",
    }


def _sentiment_value(sentiment: SentimentOutput, dimension: str) -> Intensity | None:
    """
    Extract intensity value for a sentiment dimension.

    Args:
        sentiment: The sentiment output to extract from
        dimension: Name of the dimension (satisfaction, frustration, confusion)

    Returns:
        Intensity enum value or None if invalid
    """
    value_raw = getattr(sentiment, dimension, None)
    if value_raw is None:
        return None
    try:
        return Intensity(value_raw)
    except ValueError:
        return None


def _should_trigger_escalation(sentiment: SentimentOutput) -> bool:
    """
    Determine if sentiment levels warrant escalation.

    Args:
        sentiment: The analyzed sentiment output

    Returns:
        True if any monitored dimension exceeds the trigger threshold
    """
    if not ESCALATION_ENABLED:
        return False
    for dimension in ESCALATION_DIMENSIONS:
        value = _sentiment_value(sentiment, dimension)
        if value is None:
            continue
        if INTENSITY_RANK[value] >= INTENSITY_RANK[ESCALATION_TRIGGER_INTENSITY]:
            return True
    return False


def _render_escalation_message(
    sentiment: SentimentOutput,
    user_input: str,
    metadata: dict[str, str],
) -> str:
    """
    Render the escalation notification message.

    Args:
        sentiment: The analyzed sentiment output
        user_input: The user's message text
        metadata: Session metadata (session_id, user_name)

    Returns:
        Formatted escalation message string
    """
    trimmed_input = (user_input or "").strip()
    if len(trimmed_input) > 600:
        trimmed_input = f"{trimmed_input[:600]}..."
    if not trimmed_input:
        trimmed_input = "<no user text captured>"
    sentiment_summary = (
        f"satisfaction={sentiment.satisfaction}, "
        f"frustration={sentiment.frustration}, "
        f"confusion={sentiment.confusion}"
    )
    data = {
        "user_input": trimmed_input,
        "sentiment_summary": sentiment_summary,
        "session_id": metadata.get("session_id", "<unknown session>"),
        "user_name": metadata.get("user_name", "<unknown user>"),
    }
    template = ESCALATION_MESSAGE_TEMPLATE or DEFAULT_ESCALATION_TEMPLATE
    try:
        message = template.format(**data)
    except KeyError as exc:
        logger.warning(
            "Invalid ESCALATION_MESSAGE_TEMPLATE placeholder %s; using default template.",
            exc,
        )
        message = DEFAULT_ESCALATION_TEMPLATE.format(**data)

    # Ensure session/user metadata is always included
    template_includes_session = "{session_id" in template
    template_includes_user = "{user_name" in template
    if not (template_includes_session and template_includes_user):
        metadata_block = (
            "Session ID: {session_id}\nUser name: {user_name}"
        ).format(**data)
        if metadata_block not in message:
            message = f"{message}\n{metadata_block}"

    return message


def _send_telegram_notification(message: str) -> bool:
    """
    Send escalation notification via Telegram.

    Args:
        message: The formatted message to send

    Returns:
        True if message was sent successfully, False otherwise
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(
            "Sentiment escalation triggered but Telegram credentials are missing."
        )
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }
    try:
        response = requests.post(url, data=payload, timeout=15)
    except requests.RequestException as exc:
        logger.warning("Failed to contact Telegram API: %s", exc)
        return False

    if response.status_code != 200:
        preview = response.text[:200]
        logger.warning("Telegram API error %s: %s", response.status_code, preview)
        return False

    logger.info("Human escalation sent via Telegram chat %s", TELEGRAM_CHAT_ID)
    return True


def _maybe_trigger_escalation(
    tool_context: ToolContext,
    sentiment: SentimentOutput,
    user_input: str,
) -> None:
    """
    Check sentiment and trigger escalation if thresholds are exceeded.

    Updates tool_context.state with escalation status information.

    Args:
        tool_context: The ADK tool context for state management
        sentiment: The analyzed sentiment output
        user_input: The user's message text
    """
    if tool_context.state.get("escalation:notified"):
        return
    if not _should_trigger_escalation(sentiment):
        return

    if ESCALATION_NOTICE_TEXT:
        tool_context.state["escalation_notice"] = ESCALATION_NOTICE_TEXT

    metadata = _extract_session_metadata(tool_context)
    message = _render_escalation_message(sentiment, user_input, metadata)
    success = _send_telegram_notification(message)
    tool_context.state["escalation:notified"] = True
    tool_context.state["escalation:status"] = "sent" if success else "failed"
    tool_context.state["escalation:channel"] = "telegram"


######################################
## SENTIMENT TOOL
######################################


@FunctionTool
def run_sentiment_analysis(tool_context: ToolContext) -> dict[str, Any]:
    """
    Run hybrid sentiment analysis and persist result in session state.

    Analyzes the user's emotional state using LLM + embedding fusion,
    stores the result in session state, and triggers escalation if needed.

    Args:
        tool_context: The ADK tool context for accessing user input and state

    Returns:
        Dict containing status, reason, and sentiment analysis results

    Examples:
        >>> # Called automatically by the sentiment agent
        >>> result = run_sentiment_analysis(tool_context)
        >>> print(result["sentiment"])
        {'satisfaction': 'low', 'frustration': 'high', 'confusion': 'medium'}
    """
    user_input = _extract_user_input(tool_context).strip()
    if not user_input:
        result = _neutral_sentiment_output()
        reason = "empty_input"
    elif sentiment_service is None:
        logger.warning("SentimentService unavailable; returning neutral sentiment.")
        result = _neutral_sentiment_output()
        reason = "service_unavailable"
    else:
        try:
            result = sentiment_service.analyze_emotion(user_input)
            reason = "ok"
        except RuntimeError as runtime_error:
            logger.warning("SentimentService runtime error: %s", runtime_error)
            try:
                sentiment_service.build_sentiment_phrases_embeddings(use_cache=False)
                result = sentiment_service.analyze_emotion(user_input)
                reason = "rebuilt"
            except Exception as rebuild_error:
                logger.error(
                    "Failed rebuilding sentiment prototypes: %s", rebuild_error
                )
                result = _neutral_sentiment_output()
                reason = "fallback"
        except Exception as unexpected_error:
            logger.error("Unexpected sentiment analysis error: %s", unexpected_error)
            result = _neutral_sentiment_output()
            reason = "fallback"

    result_json = result.model_dump(mode='json')
    tool_context.state["sentiment_output"] = result_json
    _maybe_trigger_escalation(tool_context, result, user_input)
    return {
        "status": "success",
        "reason": reason,
        "sentiment": result_json,
    }


logger.info("Sentiment tool configured.")
