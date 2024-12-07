import streamlit as st
import os
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.llm_manager import LLMManager
from src.query_engine import QueryEngine

# Initialize components
@st.cache_resource
def initialize_components():
    doc_processor = DocumentProcessor()
    embedding_manager = EmbeddingManager()
    llm_manager = LLMManager()
    query_engine = QueryEngine(embedding_manager, llm_manager)
    return doc_processor, embedding_manager, query_engine

def main():
    st.title("Document Analysis and Comparison Engine")
    
    # Initialize components
    doc_processor, embedding_manager, query_engine = initialize_components()
    
    # File upload section
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Save uploaded files
        for uploaded_file in uploaded_files:
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Process documents
        documents = doc_processor.process_documents()
        st.success(f"Successfully processed {len(documents)} documents")
        
        # Add to vector store
        for doc_name, content in documents.items():
            chunks = doc_processor.chunk_text(content)
            embedding_manager.add_documents({doc_name: chunks})
    
    # Query section
    st.header("Document Analysis")
    query = st.text_input("Enter your question about the documents")
    if query:
        with st.spinner("Generating response..."):
            response = query_engine.query(query)
            st.write(response)
    
    # Sample questions
    st.sidebar.header("Sample Questions")
    sample_questions = [
        "What are the risk factors associated with Google and Tesla?",
        "What is the total revenue for Google Search?",
        "What are the differences in the business of Tesla and Uber?"
    ]
    
    for question in sample_questions:
        if st.sidebar.button(question):
            with st.spinner("Generating response..."):
                response = query_engine.query(question)
                st.write(response)

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    main()
