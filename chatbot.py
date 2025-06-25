import streamlit as st
import os
import warnings
from dotenv import load_dotenv

# Import our custom modules
from llm.llm import CyberSecurityLLM, add_newline_after_bullet
from memory.memory import ChatMemory
from rag.rag import RAGRetriever, RAGPipeline

# Suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# CSS Styling
st.markdown(
    """
    <style>
    /* Latar belakang aplikasi */
    .stApp {
        background-color: #1E1E2F; /* Latar belakang abu-abu gelap */
        color: white;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;  /* Mengatur warna latar belakang tombol */
    }
    .title {
        color: white;
        font-size: 3em;
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .content {
        color: white;
        font-size: 1em;
    }
    label {
        color: white !important;
    }
    input {
        color: white !important;
        background-color: black !important; /* Jika ingin kotak input hitam */
    }
    ::placeholder { /* Untuk placeholder teks */
        color: white !important;
    }
    .st-emotion-cache-128upt6 {
        background-color: transparent !important;
    }
    .st-emotion-cache-1flajlm{
        color: white        
    }
    .st-emotion-cache-1p2n2i4 {
        height: 500px 
    }
    .st-emotion-cache-8p7l3w {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<div class="title">Chatbot - CyberSecurity</div>', unsafe_allow_html=True)

# Configuration
USER_ID = "chat_pdf"  # You can make this dynamic later
DB_PATH = "db/sqlite/chat_history.db"
VECTOR_STORE_PATH = "db/vectors"

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize and cache the RAG and LLM components."""
    try:
        # Initialize RAG retriever
        rag_retriever = RAGRetriever(vector_store_path=VECTOR_STORE_PATH)
        rag_pipeline = RAGPipeline(rag_retriever)
        
        # Initialize LLM
        llm = CyberSecurityLLM()
        
        # Initialize memory
        memory = ChatMemory(db_path=DB_PATH)
        
        return rag_pipeline, llm, memory, True
        
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None, None, False

# Initialize components
rag_pipeline, llm, memory, components_loaded = initialize_components()

# Debug sidebar
with st.sidebar:
    st.header("Debug Panel")
    
    if components_loaded:
        st.success("All components loaded successfully!")
        
        # Vector Store Info
        if st.button("Check Vector Store Status"):
            info = rag_pipeline.retriever.get_vector_store_info()
            if info['status'] == 'loaded':
                st.success("Vector store loaded successfully!")
                st.json(info)
            else:
                st.error(f"Vector store error: {info.get('error', 'Unknown error')}")
        
        # Test Retrieval
        if st.button("Test RAG System"):
            test_result = rag_pipeline.retriever.test_retrieval()
            if test_result['status'] == 'success':
                st.success("RAG system working!")
                st.json(test_result)
            else:
                st.error(f"RAG test failed: {test_result.get('error', 'Unknown error')}")
        
        # Database Status
        if st.button("Check Database Status"):
            db_status = memory.get_database_status()
            if db_status['status'] == 'connected':
                st.success("Database connected!")
                st.json(db_status)
                
                # Show message count for current user
                count = memory.get_message_count(USER_ID)
                st.info(f"Messages for user '{USER_ID}': {count}")
            else:
                st.error(f"Database error: {db_status.get('error', 'Unknown error')}")
        
        # Show Recent Messages
        if st.button("Show Recent Messages"):
            recent_messages = memory.get_recent_messages(USER_ID)
            if recent_messages:
                st.write(f"Recent messages from database:")
                for i, msg in enumerate(recent_messages):
                    st.write(f"{i+1}. {msg['role']}: {msg['content']}")
            else:
                st.write("No recent messages found.")
    
    else:
        st.error("Components failed to load!")

# Chat functionality (only if components loaded)
if not components_loaded:
    st.error("Cannot start chat - components not loaded properly. Check the debug panel for details.")
    st.stop()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = memory.load_chat_history(USER_ID)

# New conversation button
if st.button("Start New Conversation"):
    st.session_state.chat_history = []
    memory.clear_session(USER_ID)
    st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
prompt = st.chat_input("What is up?")

if prompt:
    # Update the height after user input
    st.markdown("""
        <style>
            .st-emotion-cache-1p2n2i4 {
                height: unset !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Save user message
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    memory.save_message(USER_ID, 'user', prompt)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_list = []
        
        # Get context from RAG
        context = rag_pipeline.get_context_for_query(prompt)
        
        # Get previous chat history (excluding current user message)
        previous_history = st.session_state.chat_history[:-1]
        
        # Generate response
        for chunk in llm.generate_response(prompt, context, previous_history):
            response_list.append(chunk)
            full_response = "".join(response_list)
            formatted_response = add_newline_after_bullet(full_response)
            response_container.markdown(formatted_response)

    # Save assistant response
    st.session_state.chat_history.append({'role': 'assistant', 'content': formatted_response})
    memory.save_message(USER_ID, 'assistant', formatted_response)