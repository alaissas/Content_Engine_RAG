from typing import List, Dict
import re

class LLMManager:
    def __init__(self):
        """Initialize the LLM Manager with basic text analysis capabilities."""
        pass

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text using basic text analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of key points
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter out empty sentences and clean them
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic filtering for important sentences (containing key terms)
        key_terms = ['risk', 'revenue', 'business', 'growth', 'market', 'product', 'service', 
                    'technology', 'competition', 'regulatory', 'financial']
        
        key_points = []
        for sentence in sentences:
            if any(term in sentence.lower() for term in key_terms):
                key_points.append(sentence)
        
        return key_points[:5]  # Return top 5 key points

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate a response using text analysis.
        
        Args:
            query (str): User query
            context (List[Dict]): Retrieved context from vector store
            
        Returns:
            str: Generated response
        """
        # Extract text from context
        context_text = "\n".join([c['text'] for c in context])
        
        # Extract key points
        key_points = self._extract_key_points(context_text)
        
        # Format response
        response = "Based on the available information:\n\n"
        response += "\n".join(f"- {point}" for point in key_points)
        
        return response

    def compare_documents(self, contexts: Dict[str, List[Dict]], aspect: str) -> str:
        """Compare different aspects of documents.
        
        Args:
            contexts (Dict[str, List[Dict]]): Context from different documents
            aspect (str): Aspect to compare
            
        Returns:
            str: Comparison analysis
        """
        comparison = f"Comparison of {aspect} across documents:\n\n"
        
        for doc_name, context in contexts.items():
            context_text = "\n".join([c['text'] for c in context])
            key_points = self._extract_key_points(context_text)
            
            comparison += f"{doc_name}:\n"
            comparison += "\n".join(f"- {point}" for point in key_points)
            comparison += "\n\n"
        
        return comparison
