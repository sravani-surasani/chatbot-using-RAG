
# Multi-Agent RAG System with Model Context Protocol (MCP)

A modular, agent-based Retrieval-Augmented Generation (RAG) system for intelligent, context-aware question answering over diverse document types. This project features structured agent communication via a Model Context Protocol (MCP) and a user-friendly Streamlit chat interface.

##  Features

- **Multi-format Document Upload:** Supports PDF, DOCX, PPTX, CSV, TXT, and Markdown.
- **Agentic Architecture:** Modular agents for ingestion, retrieval, and LLM response.
- **Model Context Protocol (MCP):** Structured, traceable message passing between agents.
- **Semantic Search:** Uses Sentence Transformers and FAISS for fast, meaningful retrieval.
- **Conversational Chatbot UI:** Multi-turn chat, source referencing, and debug info.

##  Setup Instructions

### 1. Install Dependencies

It’s recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- streamlit
- pandas, numpy
- sentence-transformers
- faiss-cpu,Chromadb
- openai
- PyPDF2, python-docx, python-pptx

### 2. (Optional) Set OpenAI API Key

For LLM-powered answers, set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```
Or enter it in the Streamlit sidebar when running the app.

### 3. Run the Application

```bash
streamlit run app.py
```

## Usage

1. **Upload Documents:**  
   Use the sidebar to upload PDF, DOCX, PPTX, CSV, TXT, or Markdown files.

2. **Ask Questions:**  
   Type your question in the chat input. The system will process your query using the agent pipeline.

3. **View Answers & Sources:**  
   Answers are generated using retrieved document context, with clear source references.

4. **Debug Info (Optional):**  
   Enable "Show Debug Info" in the sidebar to view recent agent message traffic.

## 🧩 Architecture Overview

- **IngestionAgent:** Parses and preprocesses uploaded documents, chunks text.
- **RetrievalAgent:** Embeds chunks, builds vector store (FAISS), retrieves relevant context.
- **LLMResponseAgent:** Forms LLM prompt with context, generates answer, references sources.
- **CoordinatorAgent:** Orchestrates the workflow and maintains conversation state.
- **MCP Message Bus:** In-memory message passing for all agent communication.

## 📝 Example MCP Message

```json
{
  "sender": "RetrievalAgent",
  "receiver": "LLMResponseAgent",
  "type": "CONTEXT_RESPONSE",
  "trace_id": "abc-123",
  "payload": {
    "top_chunks": ["..."],
    "query": "What are the KPIs?"
  }
}
```

## ⚠️ Notes

- **OpenAI API Key** is required for LLM-based answers. If not provided, a fallback response will be generated.
- **Temporary Files:** Uploaded files are stored temporarily and cleaned up after processing.
- **Performance:** For large documents or many files, initial ingestion and embedding may take a few seconds.
