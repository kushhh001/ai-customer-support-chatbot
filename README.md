# AI-Powered Customer Support & FAQ Chatbot

An intelligent NLP-driven customer support chatbot that uses dense semantic embeddings to answer user queries with contextual relevance and real-time confidence scores.

## Features
- **Semantic Understanding**: Uses Hugging Face Transformers (`all-mpnet-base-v2`) for accurate query-to-FAQ matching.
- **Text Preprocessing**: Cleans and normalizes user input using NLTK.
- **FastAPI Backend**: High-performance asynchronous API endpoints.
- **Interaction Logging**: Automatically logs all user messages, bot replies, timestamps, and confidence scores into SQLite.
- **Clean UI**: Responsive web chat interface.

## Tech Stack
- **Language**: Python 3
- **NLP / AI**: Sentence-Transformers, Hugging Face, NLTK
- **Backend**: FastAPI, Uvicorn
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript

## Getting Started

1. Clone this repository:
   ```bash
   git clone <YOUR-GITHUB-REPO-URL>
   cd chatbot-project