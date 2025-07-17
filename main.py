import streamlit as st
import asyncio

# not need in real st deployment
from dotenv import load_dotenv
load_dotenv()

from agents_logic import get_agent_response

# === Initialize Agents ===
from agents.mm_rag_agent import (
    get_mm_rag_agent,
    get_search_plugin,
)
from agents.ochestrator_agent import get_ochestrator_agent
from agents_logic import (get_agent_response)


# === Initialize Streamlit UI ===
st.set_page_config(page_title="Economic News GPT Chatbot", page_icon="💬", layout="wide")
st.title("💬 Economic News Chatbot")

# === Initialize session state for agent and memory ===
if "thread" not in st.session_state:
    st.session_state.thread = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ochestrator_agent" not in st.session_state:
    st.session_state.ochestrator_agent = None
if "pdf_rag_agent" not in st.session_state:
    st.session_state.pdf_rag_agent = None
if "pdf_search" not in st.session_state:
    st.session_state.pdf_search = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# === Async initializer ===
async def initialize_agents():
    st.session_state.pdf_rag_agent = get_mm_rag_agent()
    st.session_state.pdf_search = get_search_plugin(
        text_index_name="pdf-economic-summary",
        table_index_name="pdf-economic-summary-tables",
        image_index_name="pdf-economic-summary-images"
    )
    st.session_state.ochestrator_agent = await get_ochestrator_agent()
    st.session_state.initialized = True

# === Force sync for async init at top ===
if not st.session_state.initialized:
    asyncio.run(initialize_agents())

# === Display full history ===
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Chat input ===
user_query = st.chat_input("Ask me anything about economic reports...")

# === If user submits a query ===
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run async logic in sync context
            response, thread = asyncio.run(get_agent_response(
                user_query,
                st.session_state.thread,
                st.session_state.ochestrator_agent,
                st.session_state.pdf_rag_agent,
                st.session_state.pdf_search
            ))
            st.session_state.thread = thread
            st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})