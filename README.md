# GenAI Real Estate Assistant (URL-Based RAG + Streamlit UI) 🏠

## Overview

This project is a **Streamlit-based Retrieval-Augmented Generation (RAG) Real Estate Assistant** built using Python.

The application allows users to:

1. Provide a **real estate property listing URL**
2. Extract and process the webpage content
3. Generate embeddings from the extracted content
4. Store embeddings in a vector database
5. Ask natural language questions about the property

The assistant retrieves relevant content and generates context-aware responses using RAG.

This demonstrates dynamic web ingestion + semantic retrieval + LLM reasoning inside a Streamlit UI.

---

## Features

- Streamlit interactive UI
- Dynamic URL ingestion
- Web content extraction
- Automatic text chunking
- Vector embedding generation
- Semantic similarity search
- Context-aware RAG responses
- Real estate property Q&A

---

## Tech Stack

- Python 3.10+
- Streamlit
- RAG Architecture
- Langchain
- ChromaDB (Vector Database)
- LLM (OpenAI / Groq / Gemini depending on configuration)

---

## Project Structure

```
.
├── main.py              # Streamlit app entry point
├── rag.py               # RAG pipeline logic
├── resources/           # Supporting resources
├── requirements.txt     # Dependencies
└── README.md
```

---

## Prerequisites

- Python 3.10+
- pip

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run main.py
```

After running the command, open the local URL shown in the terminal (usually http://localhost:8501).

---

## How It Works

1. User enters a real estate listing URL
2. The system scrapes and extracts webpage content
3. Content is chunked into smaller segments
4. Chunks are converted into embeddings
5. Embeddings are stored in ChromaDB
6. User asks questions about the property
7. Relevant chunks are retrieved
8. LLM generates context-grounded answers

---

## Example Questions

- What is the price of the property?
- How many bedrooms are available?
- What amenities are mentioned?
- Is parking available?
- What is the location advantage?

---

## Environment Variables

If using API-based LLMs, create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

(Adjust according to the LLM provider used.)

---

## Notes

- The assistant answers strictly based on retrieved context from the provided URL.
- `.env`, virtual environments, and IDE files are excluded via `.gitignore`.
- This project demonstrates dynamic RAG over live web content using a UI-driven workflow.

---

## Author

Soham Pujari
