from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
import os

class EmbeddingManager:
    def __init__(self):
        """Initialize the EmbeddingManager with TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        self.documents = []
        self.metadata = []
        self.vectors = None

    def add_documents(self, documents: Dict[str, List[str]]):
        """Add documents to the vector store.
        
        Args:
            documents (Dict[str, List[str]]): Dictionary mapping document names to chunks
        """
        for doc_name, chunks in documents.items():
            self.documents.extend(chunks)
            self.metadata.extend([{"source": doc_name} for _ in chunks])
        
        if self.documents:
            self.vectors = self.vectorizer.fit_transform(self.documents)

    def query_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the vector store for similar chunks.
        
        Args:
            query (str): Query text
            n_results (int): Number of results to return
            
        Returns:
            List[Dict]: List of similar chunks with metadata
        """
        if not self.documents:
            return []
            
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadata[idx],
                "distance": 1 - similarities[idx]
            })
            
        return results
