""" VaxTalkAssistant: An AI assistant for vaccine information using RAG and sentiment analysis.
    To launch the application:
    adk web --port <PORT_NUM>
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
from google.adk.apps.app import App

# Project Imports
from src.config import load_env_variables, get_env_variable, get_env_int, get_env_list
from src.model import SentimentOutput
from src.rag.rag import RagKnowledgeBase

print("✅ Project imports loaded")

######################################
## ENV VARIABLES
######################################

load_env_variables()
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY")
print(f"✅ API key loaded")

######################################
## CONFIGURATION
######################################

project_root = Path.cwd().parent

# Model Configuration
MODEL_RAG = get_env_variable("MODEL_RAG", "gemini-2.5-flash-lite")
MODEL_SENTIMENT = get_env_variable("MODEL_SENTIMENT", "gemini-2.5-flash-lite")
MODEL_AGGREGATOR = get_env_variable("MODEL_AGGREGATOR", "gemini-2.5-flash-lite")
MODEL_SAFETY_CHECK = get_env_variable("MODEL_SAFETY_CHECK", "gemini-2.5-flash-lite")
MODEL_REFINER = get_env_variable("MODEL_REFINER", "gemini-2.5-flash-lite")

# Paths & Directories
DOC_FOLDER_PATH = project_root / get_env_variable("DOC_FOLDER_PATH", "src/Doc_vaccini")
DOC_WEB_URL_ROOT = get_env_variable("DOC_WEB_URL_ROOT", "https://www.serviziterritoriali-asstmilano.it/servizi/vaccinazioni/")
CACHE_DIR = project_root / get_env_variable("CACHE_DIR", "cache")

# Database Configuration
APP_NAME = "VaxTalkAssistant"
SQL_ASYNC_DRIVER = get_env_variable("SQL_ASYNC_DRIVER", "aiosqlite")
DB_NAME = CACHE_DIR / get_env_variable("DB_NAME", "vaxtalk_sessions.db")
DB_URL = f"sqlite+{SQL_ASYNC_DRIVER}:///{DB_NAME}"  # Local SQLite file

# RAG Configuration
RAG_MAX_PAGES = get_env_int("RAG_MAX_PAGES", 10)
RAG_MAX_DEPTH = get_env_int("RAG_MAX_DEPTH", 2)
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

rag_kb = RagKnowledgeBase(
    api_key=GOOGLE_API_KEY,
    cache_dir=CACHE_DIR
)
print("✅ Knowledge base initialized")

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
        model=MODEL_SENTIMENT,
        retry_options=retry_config
    ),
    instruction=prompt_sentiment,
    tools=[save_sentiment],
    output_key="sentiment_output",  # The result will be stored with this key.
    #output_schema=SentimentOutput,  # Define the expected output schema.
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
        model=MODEL_AGGREGATOR,
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction=prompt_aggregator,
    output_key="aggregator_output",  # This will be the final output of the entire system.
)

print("✅ aggregator_agent created.")


######################################
## SAFETY CHECK AGENT
#####################################

safety_check_prompt = """
You are a safety enforcer agent for vaccine information.
Your job is to review the final output generated by the research agents to ensure it adheres to safety and ethical guidelines.
You must check that:
1. The information provided is accurate and based on credible sources.
2. No harmful, misleading, or dangerous advice is given.
3. There are no privacy violations or sensitive data disclosures.
4. The tone is respectful and appropriate for all audiences.

Answer Guidelines:
- If the output meets all safety criteria, respond with "APPROVED".
- OTHERWISE, if there are any issues, provide specific feedback on what needs to be changed.

<response candidate>
{aggregator_output}
</response candidate>
"""

# This agent's only job is to provide feedback or the approval signal. It has no tools.
safety_check_agent = Agent(
    name="SafetyCheckAgent",
    model=Gemini(
        model=MODEL_SAFETY_CHECK,
        retry_options=retry_config
    ),
    instruction=safety_check_prompt,
    output_key="critique",  # Stores the feedback in the state.
)

######################################
## REFINER AGENT
#####################################

# This is the function that the RefinerAgent will call to exit the loop.
@FunctionTool
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}

print("✅ exit_loop function created.")

refiner_prompt = """You are an answer refiner, the last step in ensuring quality and safety for a vaccine information response.
You have access to the response of an expert assistant, as well as a critique from a safety enforcer agent.

Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
    - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.
Always aim to improve clarity, accuracy, and safety based on the critique provided.
It is crucial to keep all the provided sources and citations in your refined answer.

<response candidate>
 {aggregator_output}
</response candidate>

<critique>
{critique}
</critique>
"""

# This agent refines the answer based on the critique from the SafetyCheckAgent. It calls the exit_loop function if the critique is "APPROVED".
refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(model=MODEL_REFINER, retry_options=retry_config),
    instruction=refiner_prompt,
    output_key="aggregator_output",  # It overwrites the aggregator_output with the new, refined version.
    tools=[exit_loop],
)

print("✅ refiner_agent created.")


######################################
## WORKFLOW AGENTS
#####################################

# The ParallelAgent runs all its sub-agents simultaneously.
parallel_rag_sentiment_agent = ParallelAgent(
    name="ParallelRAGAndSentimentTeam",
    sub_agents=[sentiment_agent, rag_agent],
)

# The LoopAgent contains the agents that will run repeatedly: Safety Check -> Refiner.
answer_refinement_loop = LoopAgent(
    name="AnswerRefinementLoop",
    sub_agents=[safety_check_agent, refiner_agent],
    max_iterations=2,  # Prevents infinite loops
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator, then the answer refinement loop.
root_agent = SequentialAgent(
    name="VaccineChatbotRootAgent",
    sub_agents=[parallel_rag_sentiment_agent, aggregator_agent, answer_refinement_loop],
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

# Export root_agent for ADK web launcher
# This is required so 'adk web' can find and load the agent
__all__ = ["root_agent", "application", "runner"]
