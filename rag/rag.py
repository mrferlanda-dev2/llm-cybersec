"""
RAG module for handling vector store loading and context retrieval.
"""
import os
import warnings
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")


class RAGRetriever:
    """Handle vector store loading and context retrieval for RAG."""
    
    def __init__(self, 
                 vector_store_path: str = "db", 
                 embedding_model: str = 'nomic-embed-text',
                 base_url: str = 'http://localhost:11434'):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store_path: Path to the FAISS vector store
            embedding_model: Name of the embedding model to use
            base_url: Base URL for the embedding model service
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.base_url = base_url
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        
        # Load vector store
        self.vector_store = self._load_vector_store()
        
        # Initialize retriever
        self.retriever = self._initialize_retriever()
    
    def _load_vector_store(self) -> Optional[FAISS]:
        """Load the FAISS vector store from disk."""
        try:
            print(f"Loading vector store from {self.vector_store_path}...")
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully!")
            return vector_store
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Make sure the vector store exists and was created with compatible embeddings.")
            return None
    
    def _initialize_retriever(self):
        """Initialize the retriever with search parameters."""
        if self.vector_store is None:
            print("Cannot initialize retriever: vector store not loaded")
            return None
        
        return self.vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 3}
        )
    
    def retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context for a given query.
        
        Args:
            query: The user's question/query
            
        Returns:
            Formatted context string from retrieved documents
        """
        if self.retriever is None:
            return "No context available - vector store not loaded."
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(query)
            
            # Format the documents into context
            context = self._format_documents(docs)
            
            print(f"Retrieved {len(docs)} relevant documents")
            return context
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return "Error retrieving context from vector store."
    
    def _format_documents(self, docs: List[Document]) -> str:
        """Format retrieved documents into a single context string."""
        if not docs:
            return "No relevant context found."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            if content:
                formatted_docs.append(f"Document {i}:\n{content}")
        
        return '\n\n'.join(formatted_docs)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform similarity search and return raw documents.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of Document objects
        """
        if self.vector_store is None:
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_vector_store_info(self) -> dict:
        """Get information about the loaded vector store."""
        if self.vector_store is None:
            return {'status': 'not_loaded', 'error': 'Vector store not loaded'}
        
        try:
            # Get basic info
            info = {
                'status': 'loaded',
                'path': self.vector_store_path,
                'embedding_model': self.embedding_model,
                'base_url': self.base_url
            }
            
            # Try to get document count (this might not always work depending on FAISS version)
            try:
                info['document_count'] = self.vector_store.index.ntotal
            except:
                info['document_count'] = 'unknown'
            
            return info
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def test_retrieval(self, test_query: str = "What is cybersecurity?") -> dict:
        """Test the retrieval system with a sample query."""
        try:
            context = self.retrieve_context(test_query)
            docs = self.similarity_search(test_query, k=1)
            
            return {
                'status': 'success',
                'test_query': test_query,
                'context_length': len(context),
                'documents_retrieved': len(docs),
                'sample_context': context[:200] + "..." if len(context) > 200 else context
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and response generation."""
    
    def __init__(self, rag_retriever: RAGRetriever):
        """Initialize RAG pipeline with a retriever."""
        self.retriever = rag_retriever
    
    def get_context_for_query(self, query: str) -> str:
        """Get formatted context for a query, handling casual queries appropriately."""
        # List of casual responses that don't need retrieval
        casual_keywords = ["nothing", "okay", "ok", "thanks", "thank you", "bye", "hi", "hello", "sup", "hey"]
        query_lower = query.lower().strip()
        
        # For very casual queries, return minimal context
        if query_lower in casual_keywords or len(query_lower) < 3:
            return "Casual conversation - no specific context needed."
        
        # For actual questions, retrieve context
        return self.retriever.retrieve_context(query) 