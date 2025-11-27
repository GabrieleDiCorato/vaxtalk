"""VaxTalkAssistant: An AI assistant for vaccine information using RAG and sentiment analysis.
To launch the application:
adk web --port 42423 --session_service_uri sqlite+aiosqlite:///cache/vaxtalk_sessions.db --logo-text VaxTalkAssistant --logo-image-url https://drive.google.com/file/d/1ajO7VOLybRS6lVEKoTiBy6YrUUlY
"""

from typing import Any
from pathlib import Path
import os

import requests

# Google ADK Imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import HttpRetryOptions, Content

# Project Imports
from vaxtalk.config import load_env_variables, get_env_variable, get_env_int, get_env_list
from vaxtalk.config.logging_config import setup_logging
from vaxtalk.connectors.llm_connection_factory import LlmConnectionFactory
from vaxtalk.model import SentimentOutput, Intensity
from vaxtalk.patches.parallel_agent_patch import patch_parallel_agent
from vaxtalk.prompts import (
    RAG_AGENT_INSTRUCTION,
    SENTIMENT_AGENT_INSTRUCTION,
    DRAFT_COMPOSER_INSTRUCTION,
    SAFETY_CHECK_INSTRUCTION,
)
from vaxtalk.rag.rag_service import RagService
from vaxtalk.sentiment.sentiment_service import SentimentService


######################################
## PROJECT ROOT
######################################

# Get project root relative to this file's location
project_root = Path(__file__).resolve().parent.parent

# Setup logging before any other operations
logger = setup_logging(log_dir=project_root / "logs", log_level="INFO")
logger.info("Project imports loaded")
logger.info("Project root set to: %s", project_root)

# Monkey-patch ADK ParallelAgent merge loop to avoid OTel detach issues.
patch_parallel_agent(logger)

######################################
## ENV VARIABLES
######################################

load_env_variables(project_root / ".env")
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY")
logger.info("API key loaded")

######################################
## CONFIGURATION
######################################


# Model Configuration
MODEL_RAG = get_env_variable("MODEL_RAG", "gemini-2.5-flash-lite")
MODEL_SENTIMENT = get_env_variable("MODEL_SENTIMENT", "gemini-2.5-flash-lite")
MODEL_AGGREGATOR = get_env_variable("MODEL_AGGREGATOR", "gemini-2.5-flash-lite")
MODEL_SAFETY_CHECK = get_env_variable("MODEL_SAFETY_CHECK", "gemini-2.5-flash-lite")
MODEL_REFINER = get_env_variable("MODEL_REFINER", "gemini-2.5-flash-lite")

# Sentiment Analysis Model Configuration (separate models for embeddings and LLM)
EMBEDDING_MODEL = get_env_variable("EMBEDDING_MODEL", "text-embedding-004")
SENTIMENT_LLM_MODEL = get_env_variable("SENTIMENT_LLM_MODEL", "openrouter/mistralai/ministral-8b")

# Paths & Directories
DOC_FOLDER_PATH = project_root / get_env_variable("DOC_FOLDER_PATH", "docs")
logger.info("Document folder path set to: %s", DOC_FOLDER_PATH)
CACHE_DIR = project_root / get_env_variable("CACHE_DIR", "cache")
logger.info("Cache directory set to: %s", CACHE_DIR)
DOC_WEB_URL_ROOT = get_env_variable("DOC_WEB_URL_ROOT", None)

# Database Configuration
APP_NAME = "VaxTalkAssistant"
SQL_ASYNC_DRIVER = get_env_variable("SQL_ASYNC_DRIVER", "aiosqlite")
DB_NAME = CACHE_DIR / get_env_variable("DB_NAME", "vaxtalk_sessions.db")
DB_URL = f"sqlite+{SQL_ASYNC_DRIVER}:///{DB_NAME}"  # Local SQLite file

# RAG Configuration
RAG_MAX_PAGES = get_env_int("RAG_MAX_PAGES", 100)
RAG_MAX_DEPTH = get_env_int("RAG_MAX_DEPTH", 5)
RAG_CHUNK_SIZE = get_env_int("RAG_CHUNK_SIZE", 800)
RAG_CHUNK_OVERLAP = get_env_int("RAG_CHUNK_OVERLAP", 200)
RAG_RETRIEVAL_K = get_env_int("RAG_RETRIEVAL_K", 5)

# API Retry Configuration
RETRY_ATTEMPTS = get_env_int("RETRY_ATTEMPTS", 3)
RETRY_INITIAL_DELAY = get_env_int("RETRY_INITIAL_DELAY", 1)
RETRY_HTTP_STATUS_CODES = get_env_list("RETRY_HTTP_STATUS_CODES", [429, 500, 503, 504])

######################################
## ESCALATION CONFIGURATION
######################################

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

INTENSITY_RANK = {
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

# Clear cache to force rebuild
# rag_kb.clear_cache()

######################################
## SENTIMENT SERVICE SETUP
######################################

try:
    sentiment_service = SentimentService()
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
## AGENTS SETUP
######################################

# Configure retry settings for API calls
retry_config = HttpRetryOptions(
    attempts=RETRY_ATTEMPTS,
    initial_delay=RETRY_INITIAL_DELAY,
    http_status_codes=RETRY_HTTP_STATUS_CODES
)

######################################
## RAG AGENT
######################################

# Create retrieval function with configured k parameter
def retrieve_info(query: str) -> str:
    """
    Retrieve relevant vaccine information from the knowledge base.

    Args:
        query: The user's question about vaccines

    Returns:
        Formatted string with relevant information and sources
    """
    return rag_kb.retrieve(query, k=RAG_RETRIEVAL_K)

# Create retrieval tool from the knowledge base
rag_tool = FunctionTool(retrieve_info)

# Create the agent
rag_agent = Agent(
    name="RAG_Vaccine_Informer",
    model=LlmConnectionFactory.get_llm_connection(
        model_full_name=MODEL_RAG,
        retry_config=retry_config
    ),
    instruction=RAG_AGENT_INSTRUCTION,
    tools=[rag_tool],
    output_key="rag_output",
)

logger.info("RAG Agent configured")


######################################
## SENTIMENT AGENT
######################################


def _neutral_sentiment_output() -> SentimentOutput:
    return SentimentOutput(
        satisfaction=Intensity.LOW,
        frustration=Intensity.LOW,
        confusion=Intensity.LOW,
    )


def _extract_user_input(tool_context: ToolContext) -> str:
    """Best-effort extraction of the latest user utterance for tools."""

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
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_named_field(source: Any, candidates: tuple[str, ...]) -> str | None:
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
    value_raw = getattr(sentiment, dimension, None)
    if value_raw is None:
        return None
    try:
        return Intensity(value_raw)
    except ValueError:
        return None


def _should_trigger_escalation(sentiment: SentimentOutput) -> bool:
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



@FunctionTool
def run_sentiment_analysis(tool_context: ToolContext) -> dict[str, Any]:
    """Run hybrid sentiment analysis and persist result in session state."""

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

# There are two different storage areas:
# tool_context.state[...] is the shared session state backing the entire workflow.
# Only the tool mutates the sentiment_output entry there, and no other component writes to that key.
# The agent’s output_key controls what value gets returned as this agent’s final response
# to the orchestrator; The agent's sentiment_output does not overwrite the session state directly.
sentiment_agent = Agent(
    name="sentiment_analysis",
    model=LlmConnectionFactory.get_llm_connection(
        model_full_name=MODEL_SENTIMENT,
        retry_config=retry_config
    ),
    instruction=SENTIMENT_AGENT_INSTRUCTION,
    tools=[run_sentiment_analysis],
    output_key="sentiment_output",
   #output_schema=SentimentOutput,
)

logger.info("Sentiment agent created.")


######################################
## DRAFT COMPOSER AGENT
######################################

draft_composer_agent = Agent(
    name="DraftComposerAgent",
    model=LlmConnectionFactory.get_llm_connection(
        model_full_name=MODEL_AGGREGATOR,
        retry_config=retry_config
    ),
    instruction=DRAFT_COMPOSER_INSTRUCTION,
    output_key="draft_response",
)

logger.info("DraftComposerAgent configured")


######################################
## SAFETY CHECK AGENT
######################################

@FunctionTool
def flag_for_human_review(
    tool_context: ToolContext, reason: str, severity: str
) -> dict[str, str]:
    """
    Flag a response for human review when safety concerns are critical.

    Args:
        reason: Explanation of why human review is needed
        severity: One of ["low", "medium", "high", "critical"]
    """
    tool_context.state["flagged_for_review"] = True
    tool_context.state["flag_reason"] = reason
    tool_context.state["flag_severity"] = severity

    # In production, this could trigger a notification or queue system
    logger.warning("FLAGGED FOR REVIEW [%s]: %s", severity, reason)

    return {"status": "flagged", "severity": severity}


safety_check_agent = Agent(
    name="SafetyCheckAgent",
    model=LlmConnectionFactory.get_llm_connection(
        model_full_name=MODEL_SAFETY_CHECK,
        retry_config=retry_config
    ),
    instruction=SAFETY_CHECK_INSTRUCTION,
    tools=[flag_for_human_review],
    output_key="final_response",
)

logger.info("SafetyCheckAgent configured")


######################################
## AGGREGATOR WITH SAFETY (COMPOSITE AGENT)
######################################

# Create a sequential agent that combines draft composition with safety checking
aggregator_with_safety = SequentialAgent(
    name="AggregatorWithSafety",
    sub_agents=[
        draft_composer_agent,  # Compose draft response
        safety_check_agent,    # Validate and finalize
    ],
)

logger.info("AggregatorWithSafety configured")


######################################
## WORKFLOW AGENTS
######################################

# The ParallelAgent runs all its sub-agents simultaneously.
parallel_rag_sentiment_agent = ParallelAgent(
    name="ParallelRAGAndSentimentTeam",
    sub_agents=[sentiment_agent, rag_agent],
)

# Simplified root agent - safety is now embedded in the aggregator
root_agent = SequentialAgent(
    name="VaccineChatbotRootAgent",
    sub_agents=[
        parallel_rag_sentiment_agent,  # Parallel: RAG + Sentiment
        aggregator_with_safety,        # Sequential: Draft → Safety
    ],
)

logger.info("Root agent workflow configured")


######################################
## RUNNER SETUP
######################################

# Persistent memory using a SQLite database
# SQLite database will be created automatically
session_service = DatabaseSessionService(db_url=DB_URL)

events_compaction_config = EventsCompactionConfig(
    compaction_interval=3,  # Trigger compaction every 3 invocations
    overlap_size=1,  # Keep 1 previous turn for context
)

vax_talk_assistant = App(
    name=APP_NAME,
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)

runner = Runner(app=vax_talk_assistant, session_service=session_service)


######################################
## MAIN ENTRY POINT
######################################

def main():
    """Main entry point for launching VaxTalk web application.

    This function launches the ADK web interface with all configured parameters.
    It can be invoked via 'uv run vaxtalk' after installing the package.
    """
    import subprocess
    import sys
    import os

    # Change to project root directory (already computed at module level)
    os.chdir(project_root)

    logger.info("Launching VaxTalk Assistant...")
    logger.info("Working directory: %s", project_root)

    # Build the adk web command with all parameters
    cmd = [
        "adk", "web",
        "--port", "42423",
        "--session_service_uri", f'"{DB_URL}"',
        "--logo-text", APP_NAME,
        "--logo-image-url", "https://drive.google.com/file/d/1ajO7VOLybRS6lVEKoTiBy6YrUUlY"
    ]

    try:
        subprocess.run(" ".join(cmd), check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Error launching VaxTalk: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nVaxTalk Assistant stopped")
        sys.exit(0)


# Export root_agent for ADK web launcher
# This is required so 'adk web' can find and load the agent
__all__ = ["root_agent", "vax_talk_assistant", "runner", "main"]
