# Content Engine - Document Analysis and Comparison

This project implements a Content Engine that analyzes and compares multiple PDF documents using Retrieval Augmented Generation (RAG) techniques. It's specifically designed to process and compare Form 10-K filings from different companies.

## Features

- PDF document parsing and analysis
- Local embedding generation using sentence-transformers
- Vector storage using ChromaDB
- Local LLM integration for document analysis
- Interactive Streamlit interface for document comparison
- Chatbot interface for querying document insights

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your PDF documents in the `data` directory.

4. Run the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `src/`
  - `document_processor.py`: PDF processing and text extraction
  - `embedding_manager.py`: Document embedding generation
  - `query_engine.py`: Document querying and comparison logic
  - `llm_manager.py`: Local LLM integration
- `data/`: Directory for PDF documents
- `utils/`: Utility functions and helpers

## Usage

1. Launch the application using Streamlit
2. Upload PDF documents through the interface
3. Use the chatbot interface to ask questions about the documents
4. View comparisons and insights across different documents

## Sample Questions

1. What are the risk factors associated with Google and Tesla?
2. What is the total revenue for Google Search?
3. What are the differences in the business of Tesla and Uber?

## Requirements

- Python 3.8+
- See requirements.txt for full dependencies
