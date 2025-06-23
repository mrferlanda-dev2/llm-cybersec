#!/usr/bin/env python3
"""
Command-line interface for the cybersecurity chatbot.
Usage: python cli_chat.py [--user-id USER_ID] [--db-path DB_PATH] [--vector-store-path VECTOR_PATH]
"""

import argparse
import sys
import os
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import CyberSecurityLLM, add_newline_after_bullet
from memory.memory import ChatMemory
from rag.rag import RAGRetriever, RAGPipeline


class CLIChatbot:
    """Command-line interface for the cybersecurity chatbot."""
    
    def __init__(self, 
                 user_id: str,
                 db_path: str = "chat_history.db",
                 vector_store_path: str = "db"):
        """
        Initialize the CLI chatbot.
        
        Args:
            user_id: Unique identifier for the user session
            db_path: Path to the SQLite database for chat history
            vector_store_path: Path to the FAISS vector store
        """
        self.user_id = user_id
        print(f"Initializing chatbot for user: {user_id}")
        
        try:
            # Initialize components
            print("Loading RAG system...")
            self.rag_retriever = RAGRetriever(vector_store_path=vector_store_path)
            self.rag_pipeline = RAGPipeline(self.rag_retriever)
            
            print("Loading LLM...")
            self.llm = CyberSecurityLLM()
            
            print("Connecting to memory...")
            self.memory = ChatMemory(db_path=db_path)
            
            # Load existing chat history
            self.chat_history = self.memory.load_chat_history(user_id)
            
            print("All components initialized successfully!")
            print(f"Loaded {len(self.chat_history)} previous messages")
            
        except Exception as e:
            print(f"Failed to initialize chatbot: {e}")
            sys.exit(1)
    
    def display_chat_history(self, limit: int = 5):
        """Display recent chat history."""
        if not self.chat_history:
            print("No previous conversation history.")
            return
        
        print(f"\nRecent conversation (last {limit} messages):")
        print("-" * 50)
        
        recent_messages = self.chat_history[-limit:] if len(self.chat_history) > limit else self.chat_history
        
        for msg in recent_messages:
            role_icon = "👤" if msg['role'] == 'user' else "🤖"
            role_name = "You" if msg['role'] == 'user' else "Assistant"
            print(f"{role_icon} {role_name}: {msg['content']}")
            print()
    
    def chat_loop(self):
        """Main chat loop."""
        print("\n" + "="*60)
        print("CyberSecurity AI Assistant")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'history' to see recent conversation")
        print("Type 'clear' to start a new conversation")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! Chat history has been saved.")
                    break
                
                elif user_input.lower() == 'history':
                    self.display_chat_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.clear_conversation()
                    continue
                
                elif not user_input:
                    print("Please enter a message or 'quit' to exit.")
                    continue
                
                # Process the message
                self.process_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Chat history has been saved.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again or type 'quit' to exit.")
    
    def process_message(self, user_input: str):
        """Process a user message and generate response."""
        # Save user message
        self.chat_history.append({'role': 'user', 'content': user_input})
        self.memory.save_message(self.user_id, 'user', user_input)
        
        # Get context from RAG
        context = self.rag_pipeline.get_context_for_query(user_input)
        
        # Get previous chat history (excluding current user message)
        previous_history = self.chat_history[:-1]
        
        print("\nAssistant: ", end="", flush=True)
        
        # Generate and display response
        response_chunks = []
        try:
            for chunk in self.llm.generate_response(user_input, context, previous_history):
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
            
            # Complete response
            full_response = "".join(response_chunks)
            formatted_response = add_newline_after_bullet(full_response)
            
            # Save assistant response
            self.chat_history.append({'role': 'assistant', 'content': formatted_response})
            self.memory.save_message(self.user_id, 'assistant', formatted_response)
            
            print()  # New line after response
            
        except Exception as e:
            print(f"\nError generating response: {e}")
    
    def clear_conversation(self):
        """Clear the current conversation."""
        self.chat_history = []
        self.memory.clear_session(self.user_id)
        print("Conversation cleared. Starting fresh!")
    
    def show_system_status(self):
        """Display system status information."""
        print("\nSystem Status:")
        print("-" * 30)
        
        # Vector store info
        vector_info = self.rag_retriever.get_vector_store_info()
        print(f"Vector Store: {vector_info['status']}")
        if vector_info['status'] == 'loaded':
            print(f"Path: {vector_info['path']}")
            print(f"Documents: {vector_info.get('document_count', 'unknown')}")
        
        # Database info
        db_info = self.memory.get_database_status()
        print(f"Database: {db_info['status']}")
        if db_info['status'] == 'connected':
            print(f"Total messages: {db_info['total_messages']}")
            print(f"Unique sessions: {db_info['unique_sessions']}")
        
        # Current session info
        message_count = self.memory.get_message_count(self.user_id)
        print(f"Current session messages: {message_count}")


def main():
    """Main entry point for the CLI chatbot."""
    parser = argparse.ArgumentParser(description="CyberSecurity AI Assistant - Command Line Interface")
    parser.add_argument("--user-id", 
                       default="cli_user", 
                       help="Unique identifier for the user session (default: cli_user)")
    parser.add_argument("--db-path", 
                       default="db/sqlite/chat_history.db", 
                       help="Path to SQLite database (default: db/sqlite/chat_history.db)")
    parser.add_argument("--vector-store-path", 
                       default="db/vectors", 
                       help="Path to FAISS vector store (default: db/vectors)")
    parser.add_argument("--status", 
                       action="store_true", 
                       help="Show system status and exit")
    parser.add_argument("--history", 
                       action="store_true", 
                       help="Show chat history and exit")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = CLIChatbot(
        user_id=args.user_id,
        db_path=args.db_path,
        vector_store_path=args.vector_store_path
    )
    
    # Handle special flags
    if args.status:
        chatbot.show_system_status()
        return
    
    if args.history:
        chatbot.display_chat_history(limit=10)
        return
    
    # Start chat loop
    chatbot.chat_loop()


if __name__ == "__main__":
    main() 