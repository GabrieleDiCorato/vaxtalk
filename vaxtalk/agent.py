"""VaxTalkAssistant: An AI assistant for vaccine information using RAG and sentiment analysis.

To launch the application:
    uv run vaxtalk
"""

from pathlib import Path

# Google ADK Imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai.types import HttpRetryOptions

# Project Imports
from vaxtalk.config import load_env_variables, get_env_variable, get_env_int, get_env_list
from vaxtalk.config.logging_config import setup_logging
from vaxtalk.connectors.llm_connection_factory import LlmConnectionFactory
from vaxtalk.patches.parallel_agent_patch import patch_parallel_agent
from vaxtalk.prompts import (
    RAG_AGENT_INSTRUCTION,
    SENTIMENT_AGENT_INSTRUCTION,
    DRAFT_COMPOSER_INSTRUCTION,
    SAFETY_CHECK_INSTRUCTION,
)
from vaxtalk.tools import rag_tool, run_sentiment_analysis, flag_for_human_review, get_draft_for_validation


######################################
## PROJECT ROOT
######################################

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
logger.info("Environment variables loaded")

######################################
## CONFIGURATION
######################################

# Application Configuration
APP_NAME = get_env_variable("APP_NAME", "VaxTalkAssistant")
DEFAULT_LANGUAGE = get_env_variable("DEFAULT_LANGUAGE", "italian")
APP_LOGO = get_env_variable("APP_LOGO", None)

# Model Configuration
MODEL_RAG = get_env_variable("MODEL_RAG", "gemini-2.5-flash-lite")
MODEL_SENTIMENT = get_env_variable("MODEL_SENTIMENT", "gemini-2.5-flash-lite")
MODEL_AGGREGATOR = get_env_variable("MODEL_AGGREGATOR", "gemini-2.5-flash-lite")
MODEL_SAFETY_CHECK = get_env_variable("MODEL_SAFETY_CHECK", "gemini-2.5-flash-lite")

# Database Configuration
CACHE_DIR = project_root / get_env_variable("CACHE_DIR", "cache")
SQL_ASYNC_DRIVER = get_env_variable("SQL_ASYNC_DRIVER", "aiosqlite")
DB_NAME = CACHE_DIR / get_env_variable("DB_NAME", "vaxtalk_sessions.db")
DB_URL = f"sqlite+{SQL_ASYNC_DRIVER}:///{DB_NAME}"

# API Retry Configuration
RETRY_ATTEMPTS = get_env_int("RETRY_ATTEMPTS", 3)
RETRY_INITIAL_DELAY = get_env_int("RETRY_INITIAL_DELAY", 1)
RETRY_HTTP_STATUS_CODES = get_env_list("RETRY_HTTP_STATUS_CODES", [429, 500, 503, 504])

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

rag_agent = Agent(
    name="RAG_Vaccine_Informer",
    description="Provides vaccine-related information using a retrieval-augmented generation approach.",
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

sentiment_agent = Agent(
    name="sentiment_analysis",
    description="Analyzes user sentiment to inform response tone and content.",
    model=LlmConnectionFactory.get_llm_connection(
        model_full_name=MODEL_SENTIMENT,
        retry_config=retry_config
    ),
    instruction=SENTIMENT_AGENT_INSTRUCTION,
    tools=[run_sentiment_analysis],
    output_key="sentiment_analysis_summary",
)

logger.info("Sentiment Agent configured")

######################################
## DRAFT COMPOSER AGENT
######################################

draft_composer_agent = Agent(
    name="DraftComposerAgent",
    description="Composes draft responses based on RAG and sentiment inputs.",
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

safety_check_agent = Agent(
    name="SafetyCheckAgent",
    description="Validates draft responses for safety and compliance with medical communication standards.",
    model=LlmConnectionFactory.get_llm_connection(
        model_full_name=MODEL_SAFETY_CHECK,
        retry_config=retry_config
    ),
    instruction=SAFETY_CHECK_INSTRUCTION,
    tools=[get_draft_for_validation, flag_for_human_review],
    output_key="final_response",
)

logger.info("SafetyCheckAgent configured")

######################################
## COMPOSITE AGENTS (WORKFLOW)
######################################

# Sequential agent: draft composition followed by safety checking
aggregator_with_safety = SequentialAgent(
    name="AggregatorWithSafety",
    description="Composes draft responses and validates them for safety.",
    sub_agents=[
        draft_composer_agent,
        safety_check_agent,
    ],
)

logger.info("AggregatorWithSafety configured")

# Parallel agent: RAG and Sentiment run simultaneously
parallel_rag_sentiment_agent = ParallelAgent(
    name="ParallelRAGAndSentimentTeam",
    description="Runs RAG and Sentiment analysis agents in parallel.",
    sub_agents=[sentiment_agent, rag_agent],
)

# Root agent: orchestrates the full workflow
root_agent = SequentialAgent(
    name="VaccineChatbotRootAgent",
    description="Root agent orchestrating RAG, Sentiment Analysis, Draft Composition, and Safety Checking for vaccine information.",
    sub_agents=[
        parallel_rag_sentiment_agent,  # Parallel: RAG + Sentiment
        aggregator_with_safety,        # Sequential: Draft -> Safety
    ],
)

logger.info("Root agent workflow configured")

######################################
## RUNNER SETUP (not used when launching via ADK web)
######################################

# Persistent memory using a SQLite database
session_service = DatabaseSessionService(db_url=DB_URL)

events_compaction_config = EventsCompactionConfig(
    compaction_interval=3,  # Trigger compaction every 3 invocations
    overlap_size=1,         # Keep 1 previous turn for context
)

vax_talk_assistant = App(
    name=APP_NAME,
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)

#initial_state = {"default__response_language": DEFAULT_LANGUAGE}

#session = session_service.create_session(
#    app_name=APP_NAME, user_id="user", state=initial_state
#)
runner = Runner(app=vax_talk_assistant, session_service=session_service)

######################################
## MAIN ENTRY POINT
######################################


def main():
    """
    Main entry point for launching VaxTalk web application.

    This function launches the ADK web interface with all configured parameters.
    It can be invoked via 'uv run vaxtalk' after installing the package.
    """
    import subprocess
    import sys
    import os

    # Change to project root directory
    os.chdir(project_root)

    logger.info("Launching VaxTalk Assistant...")
    logger.info("Working directory: %s", project_root)

    cmd = [
        "adk", "web",
        "--port", "42423",
        "--session_service_uri", f'"{DB_URL}"',
        "--logo-text", APP_NAME,
    ]
    if APP_LOGO:
        cmd.extend(["--logo-image-url", APP_LOGO])

    try:
        subprocess.run(" ".join(cmd), check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Error launching VaxTalk: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nVaxTalk Assistant stopped")
        sys.exit(0)


# Export root_agent for ADK web launcher
__all__ = ["root_agent", "vax_talk_assistant", "runner", "main"]
