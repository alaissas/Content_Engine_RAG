from typing import List, Dict
from .embedding_manager import EmbeddingManager
from .llm_manager import LLMManager

class QueryEngine:
    def __init__(self, embedding_manager: EmbeddingManager, llm_manager: LLMManager):
        """Initialize the Query Engine.
        
        Args:
            embedding_manager (EmbeddingManager): Instance of EmbeddingManager
            llm_manager (LLMManager): Instance of LLMManager
        """
        self.embedding_manager = embedding_manager
        self.llm_manager = llm_manager

    def query(self, query: str, n_results: int = 5) -> str:
        """Process a query and generate a response.
        
        Args:
            query (str): User query
            n_results (int): Number of similar chunks to retrieve
            
        Returns:
            str: Generated response
        """
        # Retrieve relevant context
        similar_chunks = self.embedding_manager.query_similar(query, n_results)
        
        # Generate response using LLM
        response = self.llm_manager.generate_response(query, similar_chunks)
        return response

    def compare_documents(self, query: str, doc_names: List[str], n_results: int = 3) -> str:
        """Compare specific aspects across documents.
        
        Args:
            query (str): Comparison query
            doc_names (List[str]): Names of documents to compare
            n_results (int): Number of chunks per document
            
        Returns:
            str: Comparison analysis
        """
        # Get relevant chunks for each document
        contexts = {}
        for doc_name in doc_names:
            similar_chunks = self.embedding_manager.query_similar(
                query,
                n_results,
                lambda x: x['metadata']['source'] == doc_name
            )
            contexts[doc_name] = similar_chunks
        
        # Generate comparison using LLM
        comparison = self.llm_manager.compare_documents(contexts, query)
        return comparison
