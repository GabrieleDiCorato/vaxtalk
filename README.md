# VaxTalk - AI-Powered Vaccine Information Assistant

A Retrieval-Augmented Generation (RAG) chatbot that provides accurate vaccine information by combining document retrieval with AI-powered responses and sentiment analysis.

## 📋 Context

This project was developed as a **homework assignment for a Master's program in AI**, completed within one week of part-time work. The goal was to create a practical application demonstrating:

- **RAG (Retrieval-Augmented Generation)** techniques for grounding AI responses in factual documents
- **Multi-agent orchestration** using Google's Agent Development Kit (ADK)
- **Sentiment analysis** to understand user emotional state
- **Document processing** from multiple sources (PDFs and web pages)
- **Vector embeddings** for semantic search
- **Persistent session management** for conversational continuity. An introduction to context engineering.

## 🎯 Project Overview

VaxTalk assists users with vaccine-related questions by:

1. **Retrieving** relevant information from official vaccine documentation and websites
2. **Analyzing** user sentiment to provide empathetic responses, and escalate to human interaction if necessary
3. **Synthesizing** information through a multi-agent workflow
4. **Maintaining** conversation history during the session

## 📡 Human Escalation Notifications

When a conversation shows **high frustration or confusion**, VaxTalk can notify a human via Telegram and append a user-facing notice so people know someone is joining.

1. Enable the flow by setting `ESCALATION_ENABLED=true` in `.env`.
2. Provide `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (direct chat or group ID).
3. Tune thresholds with `ESCALATION_DIMENSIONS` and `ESCALATION_TRIGGER_LEVEL` (default: high frustration/confusion).
4. Customize the Telegram payload and UI notice via `ESCALATION_MESSAGE_TEMPLATE` and `ESCALATION_NOTICE_TEXT`.

If the Telegram call fails, the app logs a warning but continues responding; no retries are attempted for the same session.

## 🚀 Setup Guide

### Prerequisites

- **Python 3.11+**
- **UV** package manager (recommended) or pip
- **Google API Key** (for Gemini models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/GabrieleDiCorato/vaxtalk.git
   cd vaxtalk
   ```

2. **Install dependencies**

   Using UV (recommended):
   ```bash
   uv sync --all-groups
   ```

3. **Configure environment variables**

   Copy the `.env.example` file to create your own `.env` file:
   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file with your configuration. See `.env.example` for detailed documentation and examples of all available options.

4. **Prepare document sources**

   Place your PDF documents in the folder specified by `DOC_FOLDER_PATH` (e.g., `docs/`) and configure `DOC_WEB_URL_ROOT` if you want to crawl a website.

## 🔄 Loading the Corpus

**Before first run** or when you want to refresh the knowledge base with updated documents:

```bash
uv run load-corpus
```

This command will:
- Clear the existing cache
- Reload all PDF documents from your document folder
- Crawl the configured website (if `DOC_WEB_URL_ROOT` is set)
- Build new embeddings and save them to cache

The process may take several minutes depending on:
- Number of PDF files
- Website size and depth
- Your internet connection speed

**When to reload the corpus:**
- After adding or updating PDF documents
- When the source website has been updated
- If you change RAG configuration parameters (chunk size, overlap, etc.)
- To troubleshoot cache-related issues

## 🎮 Running the Application

### Using UV (Recommended)

The easiest way to launch VaxTalk is using the built-in uv script:

```bash
uv run vaxtalk
```

This will:
- Initialize the knowledge base
- Start the web server on port 42423
- Open the interface at `http://localhost:42423`

### Using ADK CLI Directly

Alternatively, you can use the ADK command directly:

```bash
adk web --port 42423 --session_service_uri sqlite+aiosqlite:///cache/vaxtalk_sessions.db --logo-text VaxTalkAssistant
```

Navigate to the URL provided by the adk message.

## 🧩 Importing VaxTalk Programmatically

Importing the top-level `vaxtalk` package intentionally has almost no side
effects.  When you need the ADK objects programmatically (e.g.
`from vaxtalk import root_agent`), the package now lazy-loads
`vaxtalk.agent` the moment one of those attributes is accessed.  This keeps
utility scripts such as `uv run load-corpus` lightweight while still exposing
the same public interface for ADK launchers.

## 📊 First Run

On first launch, the system will:
1. Check for existing cache in the `CACHE_DIR` folder
2. If no cache exists, automatically build the knowledge base:
   - Process PDF documents from `DOC_FOLDER_PATH`
   - Crawl website from `DOC_WEB_URL_ROOT` (if configured)
   - Generate embeddings and save to cache
3. Initialize the SQLite session database
4. Start the web server

**Note:** The initial knowledge base building may take several minutes if no cache exists. Subsequent starts will be fast, loading from the cached embeddings. Use `uv run load-corpus` to manually rebuild the cache when needed.


## 📝 Disclaimer

This project is for educational purposes as part of a Master's program coursework. It is provided as-is and it will not be maintained or updated.
