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

   Create a `.env` file in the project root. You can duplicate the `.env.template` file and create your custom configuration.
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

4. **Prepare document sources** (optional)

   Place PDF documents in `src/Doc_vaccini/` and configure the web URL.

### Running the Application Using ADK CLI

```bash
adk web --port <PORT_NUMBER>
```

Navigate to the URL provided by the adk message.

### First Run

On first launch, the system will:
1. Load or create the vector database cache
2. Process documents from PDFs and web sources (if cache doesn't exist)
3. Initialize the SQLite session database
4. Start the web server

The knowledge base building may be slow on first run, but subsequent starts will be fast using the cache.


## 📝 Disclaimer

This project is for educational purposes as part of a Master's program coursework. It is provided as-is and it will not be maintained or updated.
