import PyPDF2
from typing import List, Dict
import os

class DocumentProcessor:
    def __init__(self, data_dir: str = "data"):
        """Initialize the DocumentProcessor.
        
        Args:
            data_dir (str): Directory containing PDF documents
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return ""
        return text

    def process_documents(self) -> Dict[str, str]:
        """Process all PDF documents in the data directory.
        
        Returns:
            Dict[str, str]: Dictionary mapping document names to their content
        """
        documents = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.data_dir, filename)
                content = self.read_pdf(file_path)
                if content:
                    documents[filename] = content
        return documents

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into smaller chunks for processing.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 for space
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
