# VaxTalk Copilot Instructions

## Project Overview

VaxTalk is a **multi-agent RAG chatbot** built with **Google ADK (Agent Development Kit)** for vaccine information. The architecture combines retrieval-augmented generation with sentiment analysis through a parallel→sequential agent workflow.

**Key insight**: This is an educational project (Master's coursework) with fixed scope - focus on understanding existing patterns rather than major refactoring.

## Architecture: Multi-Agent Workflow

The agent orchestration in `vaxtalk/agent.py` follows this pattern:

```
ParallelAgent (RAG + Sentiment)
    ↓
SequentialAgent (Draft Composer → Safety Check)
    ↓
Final Response
```

**Critical design decisions**:
- **Parallel execution**: RAG retrieval and sentiment analysis run simultaneously to minimize latency
- **Sequential safety**: Draft composition must complete before safety validation
- **Session state sharing**: Agents communicate via `tool_context.state` (see `save_sentiment` tool)
- **ADK runner**: Uses SQLite session persistence at `cache/vaxtalk_sessions.db`

## Core Components

### 1. RAG Knowledge Base (`vaxtalk/rag/`)
- **No ADK dependencies** - pure Python service for reusability
- Caches embeddings as pickle files in `cache/` directory
- Combines PDF processing (`pdf_handler.py`) + web crawling (`web_handler.py`)
- Retrieval uses cosine similarity via numpy, not a vector database

**Key pattern**: `build_knowledge_base()` checks cache first, rebuilds only if empty or explicitly cleared via `load-corpus` command.

### 2. Sentiment Analysis (`vaxtalk/sentiment/`)
- **Hybrid approach**: Fuses LLM predictions with embedding similarity
- Loads emotion prototypes from `sentiment_phrases.json` at startup
- Three dimensions: `satisfaction`, `frustration`, `confusion` (each: low/medium/high)
- Configurable fusion weights via `SENTIMENT_W_LLM` and `SENTIMENT_W_EMB` env vars

**Critical**: Uses OpenRouter API for both embeddings and LLM (not Google models) - see `_eval_normalized_embeddings()` and `_call_llm_for_sentiment()`.

### 3. Configuration (`vaxtalk/config/`)
- All settings via `.env` file (copy from `.env.example`)
- Environment helpers: `get_env_variable()`, `get_env_int()`, `get_env_list()`
- Logging configured centrally in `logging_config.py` - logs to `logs/` directory

### 4. LLM Connections (`vaxtalk/connectors/`)
- Factory pattern: `LlmConnectionFactory.get_llm_connection()`
- Supports two types: `gemini-*` models (Google) or `openrouter/*` models (via LiteLLM)
- **Important**: Model names must include proper prefixes to route correctly

## Data Models (Pydantic)

All models in `vaxtalk/model/` use Pydantic v2:
- `DocumentChunk`: Includes `doc_type` enum (web/pdf) - check backward compatibility for cached chunks
- `SentimentOutput`: Uses `Intensity` enum with `use_enum_values=True` config
- Models are **frozen** and **forbid extra fields** for strict validation

## Development Workflows

### Running the Application
```bash
# Standard launch (uses ADK web server)
uv run vaxtalk

# Alternative: Direct ADK CLI
adk web --port 42423 --session_service_uri "sqlite+aiosqlite:///cache/vaxtalk_sessions.db" --logo-text VaxTalkAssistant
```

### Rebuilding Knowledge Base
```bash
# Force fresh rebuild (clears cache, reprocesses all docs)
uv run load-corpus

# Triggered automatically on first run if cache/ is empty
```

### Dependency Management
- Uses **UV** package manager (not pip)
- `uv sync --all-groups` installs all dependencies including dev tools
- Dev dependencies: Jupyter notebooks, pre-commit hooks

### Testing/Prototyping
- Notebooks in `notebooks/` for experimentation
- `sentiment_tracker/` has standalone usage notebook
- No formal test suite - this is coursework, not production

## Common Patterns

### Agent Tool Creation
```python
@FunctionTool
def my_tool(tool_context: ToolContext, param: str) -> dict:
    """Tools must return dict, can access/modify tool_context.state"""
    tool_context.state["key"] = "value"
    return {"status": "success"}
```

### Path Resolution
Always resolve paths relative to `project_root`:
```python
project_root = Path(__file__).resolve().parent.parent
DOC_FOLDER_PATH = project_root / get_env_variable("DOC_FOLDER_PATH", "docs")
```

### Error Handling in Retrieval
Return informative strings, not exceptions:
```python
if not self.chunks:
    return "Error: Knowledge base not initialized."
```

## Caveats & Constraints

1. **OpenRouter dependency**: Sentiment analysis currently requires OpenRouter API key
2. **Cache invalidation**: No automatic detection of document changes - must run `load-corpus` manually
3. **Single-session debugging**: ADK web interface doesn't support multi-user concurrent sessions well
4. **Model limitations**: Only tested with Gemini Flash and Ministral-8b - other models may need prompt tuning
5. **No tests**: Educational project - validate manually via notebooks or web interface

## File Organization Conventions

- `vaxtalk/`: Main package (importable)
- `notebooks/`: Prototypes and experiments (not production code)
- `cache/`, `logs/`, `docs/`: Runtime directories (gitignored)

## When Adding Features

- **New agent**: Add to `agent.py` sequential/parallel chains, export in `__all__`
- **New data model**: Create in `vaxtalk/model/`, use Pydantic v2 with strict validation
- **New RAG source**: Extend handlers in `vaxtalk/rag/`, keep ADK-independent
- **Configuration**: Add to `.env.example` with documentation, use type-safe helpers

## Code Style & Conventions

### Type Hints
- Use **modern Python 3.10+ type hints**: `dict[str, Any]`, `list[str]`, `str | None`
- **Avoid legacy typing imports**: `Dict`, `List`, `Optional` from `typing` module

### Communication Style
- **No emojis** in code comments, docstrings, or log messages
- Keep logging messages professional and concise
- Use clear, descriptive variable and function names without abbreviations

### Docstrings
- Follow Google-style docstrings for consistency with existing code
- Include `Args`, `Returns`, `Raises` and `Examples` sections where applicable, expecially for public functions and FunctionTools.

## Key Entry Points

- Application launch: `vaxtalk/agent.py:main()`
- Corpus rebuild: `vaxtalk/load_corpus_and_data.py:main()`
- Sentiment analysis: `vaxtalk/sentiment/sentiment_service.py:SentimentService`
- RAG retrieval: `vaxtalk/rag/rag_service.py:RagService.retrieve()`
