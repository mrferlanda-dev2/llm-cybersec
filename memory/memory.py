"""
Memory module for handling chat history storage and retrieval.
"""
import sqlite3
from typing import List, Dict, Optional
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class ChatMemory:
    """Handle chat history storage and retrieval using SQLite."""
    
    def __init__(self, db_path: str = "chat_history.db"):
        """Initialize chat memory with database path."""
        self.db_path = db_path
    
    def get_session_history(self, session_id: str) -> SQLChatMessageHistory:
        """Get SQLChatMessageHistory for a specific session."""
        return SQLChatMessageHistory(session_id, f"sqlite:///{self.db_path}")
    
    def save_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Save a single message to the database.
        
        Args:
            session_id: Unique identifier for the chat session
            role: Either 'user' or 'assistant'
            content: The message content
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            history = self.get_session_history(session_id)
            
            if role == 'user':
                history.add_message(HumanMessage(content=content))
            elif role == 'assistant':
                history.add_message(AIMessage(content=content))
            else:
                raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")
            
            return True
            
        except Exception as e:
            print(f"Error saving {role} message to database: {e}")
            return False
    
    def load_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Load existing chat history from database.
        
        Args:
            session_id: Unique identifier for the chat session
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        try:
            history = self.get_session_history(session_id)
            messages = []
            
            for message in history.messages:
                if hasattr(message, 'content'):
                    role = 'user' if message.type == 'human' else 'assistant'
                    messages.append({'role': role, 'content': message.content})
            
            print(f"Loaded {len(messages)} messages from database")
            return messages
            
        except Exception as e:
            print(f"Error loading chat history from database: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages for a specific session.
        
        Args:
            session_id: Unique identifier for the chat session
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            history = self.get_session_history(session_id)
            history.clear()
            print(f"Cleared chat history for session: {session_id}")
            return True
            
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            return False
    
    def get_message_count(self, session_id: str) -> int:
        """
        Get the number of messages for a specific session.
        
        Args:
            session_id: Unique identifier for the chat session
            
        Returns:
            int: Number of messages in the session
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM message_store WHERE session_id = ?", (session_id,))
                count = cursor.fetchone()[0]
                return count
                
        except Exception as e:
            print(f"Error getting message count: {e}")
            return 0
    
    def get_database_status(self) -> Dict[str, any]:
        """
        Get database status information.
        
        Returns:
            Dictionary with database status information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [table[0] for table in cursor.fetchall()]
                
                # Get total message count
                cursor.execute("SELECT COUNT(*) FROM message_store;")
                total_messages = cursor.fetchone()[0]
                
                # Get unique sessions
                cursor.execute("SELECT COUNT(DISTINCT session_id) FROM message_store;")
                unique_sessions = cursor.fetchone()[0]
                
                return {
                    'status': 'connected',
                    'tables': tables,
                    'total_messages': total_messages,
                    'unique_sessions': unique_sessions
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_recent_messages(self, session_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Get recent messages for debugging purposes.
        
        Args:
            session_id: Unique identifier for the chat session
            limit: Maximum number of messages to return
            
        Returns:
            List of recent message dictionaries
        """
        try:
            history = self.get_session_history(session_id)
            messages = history.messages[-limit:] if len(history.messages) > limit else history.messages
            
            result = []
            for msg in messages:
                role = "👤 User" if msg.type == 'human' else "🤖 Assistant"
                result.append({
                    'role': role,
                    'content': msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                })
            
            return result
            
        except Exception as e:
            print(f"Error getting recent messages: {e}")
            return [] 