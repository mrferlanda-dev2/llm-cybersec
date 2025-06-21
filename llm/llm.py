"""
LLM module for handling inference with ChatOllama.
"""
import os
import warnings
from typing import Iterator, List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")


class CyberSecurityLLM:
    """LLM handler for cybersecurity question-answering tasks."""
    
    def __init__(self, model_name: str = 'llama3.2:latest', base_url: str = 'http://localhost:11434'):
        """Initialize the LLM with specified model and base URL."""
        # If OLLAMA_BASE_URL is set, use it, otherwise use the base_url
        
        self.llm = ChatOllama(model=model_name, base_url=os.getenv('OLLAMA_BASE_URL', base_url))
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for cybersecurity Q&A."""
        prompt_text = """
You are an AI assistant for cyber security question-answering tasks. Use the following pieces of retrieved context and conversation history to answer the question.

IMPORTANT INSTRUCTIONS:
1. When the user asks follow-up questions or refers to previous answers, use the conversation history to understand what they're referring to.
2. If the user just says casual things like "hi", "nothing", "okay", "thanks", etc., respond naturally without forcing cyber security context.
3. Only use the retrieved context when the user asks an actual cyber security question.
4. If you don't know the answer based on the context and conversation history, just say that you don't know.
5. Be conversational and natural - don't force technical explanations when they're not needed.

Previous Conversation:
{chat_history}

Current Question: {question} 

Retrieved Context: {context} 

Answer:
"""
        return ChatPromptTemplate.from_template(prompt_text)
    
    def format_chat_history(self, chat_history: List[Dict[str, str]], max_messages: int = 6) -> str:
        """Format chat history for the LLM prompt."""
        if not chat_history:
            return "No previous conversation."
        
        # Take the last N messages to maintain context but avoid token limits
        recent_history = chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history
        formatted_history = ""
        
        for msg in recent_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n\n"
        
        return formatted_history
    
    def generate_response(self, 
                         question: str, 
                         context: str, 
                         chat_history: List[Dict[str, str]] = None) -> Iterator[str]:
        """Generate streaming response from the LLM."""
        formatted_history = self.format_chat_history(chat_history or [])
        
        # Create the chain
        chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Prepare input
        chain_input = {
            "question": question,
            "context": context,
            "chat_history": formatted_history
        }
        
        # Stream the response
        for chunk in chain.stream(chain_input):
            yield chunk
    
    def generate_response_sync(self, 
                              question: str, 
                              context: str, 
                              chat_history: List[Dict[str, str]] = None) -> str:
        """Generate synchronous response from the LLM."""
        response_chunks = list(self.generate_response(question, context, chat_history))
        return "".join(response_chunks)


def add_newline_after_bullet(text: str) -> str:
    """Add newlines after bullet points for better formatting."""
    return text.replace("•", "\n• ") 