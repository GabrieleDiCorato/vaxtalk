from pathlib import Path

# Google ADK Imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.apps.app import App

# Project Imports
from src.config import load_env_variables, get_env_variable
from src.model import SentimentOutput
from src.rag.rag import RagKnowledgeBase

print("✅ Project imports loaded")

######################################
## CONFIGURATION
######################################

project_root = Path.cwd().parent
DOC_FOLDER_PATH = project_root / "src" / "Doc_vaccini"
DOC_WEB_URL_ROOT = "https://www.serviziterritoriali-asstmilano.it/servizi/vaccinazioni/"
CACHE_DIR = project_root / "cache"

APP_NAME = "VaxTalkAssistant"
SQL_ASYNC_DRIVER = "aiosqlite"
DB_NAME = CACHE_DIR / "vaxtalk_sessions.db"
DB_URL = f"sqlite+{SQL_ASYNC_DRIVER}:///{DB_NAME}"  # Local SQLite file

######################################
## ENV VARIABLES
######################################

load_env_variables()
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY")
print(f"✅ API key loaded")

######################################
## RAG KNOWLEDGE BASE SETUP
######################################

rag_kb = RagKnowledgeBase(
    api_key=GOOGLE_API_KEY,
    cache_dir=CACHE_DIR
)
print("✅ Knowledge base initialized")

# Build knowledge base from PDFs and website
rag_kb.build_knowledge_base(
    pdf_folder=DOC_FOLDER_PATH,
    root_url=DOC_WEB_URL_ROOT,
    max_pages=10,
    max_depth=2,
    use_cache=True,
)

# Display statistics
stats = rag_kb.get_stats()
print(f"\n📊 Knowledge Base Stats:")
print(f"  Chunks: {stats['num_chunks']}")
print(f"  Embedding shape: {stats['embedding_shape']}")

# Clear cache to force rebuild
# rag_kb.clear_cache()

######################################
## AGENTS SETUP
######################################

# Configure retry settings for API calls
retry_config = types.HttpRetryOptions(
    attempts=3,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

######################################
## RAG AGENT
######################################

# Create retrieval tool from the knowledge base
rag_tool = FunctionTool(rag_kb.retrieve)

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
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=prompt_rag,
    tools=[rag_tool],
    output_key="rag_output",
)

print("✅ RAG Agent configured")


######################################
## SENTIMENT AGENT
######################################


# Test: we could use a tool to save sentiment in session state.
@FunctionTool
def save_sentiment(
    tool_context: ToolContext, sentiment: SentimentOutput
) -> dict[str, str]:
    """
    Tool to record and save the sentiment analysis result into session state.

    Args:
        sentiment (SentimentOutput): The sentiment analysis result to be saved.
    """
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["sentiment"] = sentiment

    return {"status": "success"}


print("✅ Sentiment Tools created.")

prompt_sentiment = """You are a sentiment analysis assistant."""

sentiment_agent = Agent(
    name="sentiment_analysis",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=prompt_sentiment,
    tools=[save_sentiment],
    output_key="sentiment_output",  # The result will be stored with this key.
    output_schema=SentimentOutput,  # Define the expected output schema.
)

print("✅ sentiment_agent created.")


######################################
## AGGREGATOR AGENT
######################################

prompt_aggregator = """You are an expert assistant, running a chat help line for vaccine information.
Your task is to synthesize information. You have access to the following inputs:

<User query>
{{session.state['user:input']}}
</User query>

<RAG Knowledge Base Output>
{rag_output}
</RAG Knowledge Base Output>


When replying to the User, consider that we performed a sentiment analysis on their query, with the following result:
<Sentiment Analysis Result>
{sentiment_output}
</Sentiment Analysis Result>
"""

# The AggregatorAgent runs *after* the parallel step to synthesize the results.
aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction=prompt_aggregator,
    output_key="final_output",  # This will be the final output of the entire system.
)

print("✅ aggregator_agent created.")

######################################
## WORKFLOW AGENTS
#####################################

# The ParallelAgent runs all its sub-agents simultaneously.
parallel_rag_sentiment_agent = ParallelAgent(
    name="ParallelRAGAndSentimentTeam",
    sub_agents=[sentiment_agent, rag_agent],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_rag_sentiment_agent, aggregator_agent],
)

print("✅ Parallel and Sequential Agents created.")


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

application = App(
    name=APP_NAME,
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)

runner = Runner(app=application, session_service=session_service)
