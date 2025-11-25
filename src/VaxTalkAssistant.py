"""VaxTalkAssistant: An AI assistant for vaccine information using RAG and sentiment analysis.
To launch the application:
adk web --port 42423 --session_service_uri sqlite+aiosqlite:///cache/vaxtalk_sessions.db --logo-text VaxTalkAssistant --logo-image-url https://drive.google.com/file/d/1ajO7VOLybRS6lVEKoTiBy6YrUUlY
"""

from pathlib import Path

# Google ADK Imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Project Imports
from vaxtalk.config import load_env_variables, get_env_variable, get_env_int, get_env_list
from vaxtalk.config.logging_config import setup_logging, get_logger
from vaxtalk.model import SentimentOutput
from vaxtalk.rag.rag_service import RagService


######################################
## PROJECT ROOT
######################################

# Get project root relative to this file's location
project_root = Path(__file__).resolve().parent.parent

# Setup logging before any other operations
logger = setup_logging(log_dir=project_root / "logs", log_level="INFO")
logger.info("Project imports loaded")
logger.info("Project root set to: %s", project_root)

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
## AGENTS SETUP
######################################

# Configure retry settings for API calls
retry_config = types.HttpRetryOptions(
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

# Define agent instruction
prompt_rag = """You are a helpful assistant for vaccine information.
You have access to a knowledge base containing official documents and web pages about vaccinations.

When the user asks a question:
1. Use the `retrieve` tool to find relevant information.
2. Answer the question based ONLY on the information returned by the tool.
3. If the tool returns no information, or the information is not pertinent, say you don't have that information.
4. Always cite the sources provided in the tool output.
5. Be concise but thorough in your responses.
"""

# Create the agent
rag_agent = Agent(
    name="RAG_Vaccine_Informer",
    model=Gemini(
        model=MODEL_RAG,
        retry_options=retry_config
    ),
    instruction=prompt_rag,
    tools=[rag_tool],
    output_key="rag_output",
)

logger.info("RAG Agent configured")


######################################
## SENTIMENT AGENT
######################################


# Test: we could use a tool to save sentiment in session state.
@FunctionTool
def save_sentiment(
    tool_context: ToolContext, sentiment: str
) -> dict[str, str]:
    """
    Tool to record and save the sentiment analysis result into session state.

    Args:
        sentiment (str): The sentiment analysis result to be saved.
    """

    tool_context.state["sentiment"] = sentiment
    return {"status": "success"}


logger.info("Sentiment Tools created.")

prompt_sentiment = """You are an expert at analyzing user sentiment based on their queries.
Your task is to evaluate the user's input and determine their sentiment regarding vaccine information.

<User query>
{{session.state['user:input']}}
</User query>

If the user is particularly frustrated, confused, anxious, or dissatisfied about vaccination information,
flag these sentiments using the save_sentiment tool.
"""

sentiment_agent = Agent(
    name="sentiment_analysis",
    model=Gemini(
        model=MODEL_SENTIMENT,
        retry_options=retry_config
    ),
    instruction=prompt_sentiment,
    tools=[save_sentiment],
    output_key="sentiment_output",  # The result will be stored with this key.
    #output_schema=SentimentOutput,  # Define the expected output schema.
)

logger.info("Sentiment agent created.")


######################################
## DRAFT COMPOSER AGENT
######################################

draft_composer_prompt = """You are an expert assistant for vaccine information.
Your task is to compose a comprehensive draft response.

<User query>
{{session.state['user:input']}}
</User query>

<RAG Knowledge Base Output>
{rag_output}
</RAG Knowledge Base Output>

{{#if sentiment_output}}
<User Sentiment Analysis>
{sentiment_output?}
</User Sentiment Analysis>

Adapt your response to user sentiment:
- High frustration → Acknowledge concerns explicitly and be extra clear
- High confusion → Break down into simpler terms with examples
- Low satisfaction → Provide additional reassurance and resources
- High anxiety → Emphasize consulting healthcare providers
{{/if}}

Compose a response that:
1. Answers based ONLY on the RAG output
2. Includes all source citations that the RAG output provided. Do not invent new ones.
3. Be empathetic and professional in tone

This draft will be reviewed for safety before delivery.
"""

draft_composer_agent = Agent(
    name="DraftComposerAgent",
    model=Gemini(
        model=MODEL_AGGREGATOR,
        retry_options=retry_config
    ),
    instruction=draft_composer_prompt,
    output_key="draft_response",
)

logger.info("DraftComposerAgent configured")


######################################
## SAFETY CHECK AGENT
######################################

safety_check_prompt = """
You are a safety validator for vaccine information responses.
Your job is to review the draft response and either approve it or provide a corrected version.

<User query>
{{session.state['user:input']}}
</User query>

<RAG Knowledge Base Output>
{rag_output}
</RAG Knowledge Base Output>

<Draft Response>
{draft_response}
</Draft Response>

Validate the response against these criteria:
1. Accuracy based on credible sources from RAG output
2. No harmful, misleading, or dangerous medical advice
3. No privacy violations or sensitive data disclosures
4. Respectful and appropriate tone for all audiences
5. All source citations are preserved

If the response passes all criteria:
- Return it exactly as-is

If there are issues:
- Fix them while maintaining source citations and accuracy
- If issues are critical, use the flag_for_human_review tool

Output only the final safe response text.
"""


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
    model=Gemini(
        model=MODEL_SAFETY_CHECK,
        retry_options=retry_config
    ),
    instruction=safety_check_prompt,
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
