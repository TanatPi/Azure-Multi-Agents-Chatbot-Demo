import streamlit as st
import asyncio
import time 

# not need in real st deployment
from dotenv import load_dotenv
load_dotenv()

from agents_logic import get_agent_response

# === Initialize Agents ===
from agents.mm_rag_agent import (
    get_mm_rag_agent,
    get_search_plugin,
)
from agents.orchestrator_agent import get_orchestrator_agent
from agents.keyword_extractor_agent import get_keyword_extractor_agent
from agents_logic import (get_agent_response)



# === Initialize Streamlit UI ===
st.set_page_config(page_title="Economic News GPT Chatbot", page_icon="💬", layout="wide")
st.title("💬 Economic News Chatbot")

# === Initialize session state for agent and memory ===
if "thread" not in st.session_state:
    st.session_state.thread = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "orchestrator_agent" not in st.session_state:
    st.session_state.orchestrator_agent = None
if "pdf_rag_agent" not in st.session_state:
    st.session_state.pdf_rag_agent = None
if "keyword_extractor_agent" not in st.session_state:
    st.session_state.keyword_extractor_agent = None
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
    st.session_state.keyword_extractor_agent = get_keyword_extractor_agent()
    st.session_state.orchestrator_agent = await get_orchestrator_agent()
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
            start_time = time.time()

            # Run async logic in sync context
            response, thread, rag_prompt, rag_completion, orch_prompt, orch_completion, keyword_prompt, keyword_completion   = asyncio.run(
                get_agent_response(
                    user_query,
                    st.session_state.thread,
                    st.session_state.orchestrator_agent,
                    st.session_state.pdf_rag_agent,
                    st.session_state.keyword_extractor_agent,
                    st.session_state.pdf_search
                )
            )

            end_time = time.time()
            total_time = end_time - start_time

            st.session_state.thread = thread
            st.markdown(response)
            st.markdown(f"⏱️ *Response generated in {total_time:.2f} seconds*")
            st.markdown(f"📊 *Keyword Extractor tokens: {keyword_prompt} prompt + {keyword_completion} completion = {keyword_prompt + keyword_completion} total*")
            st.markdown(f"📊 *RAG tokens: {rag_prompt} prompt + {rag_completion} completion = {rag_prompt + rag_completion} total*")
            st.markdown(f"📊 *Orchestrator tokens: {orch_prompt} prompt + {orch_completion} completion = {orch_prompt + orch_completion} total*")

    st.session_state.chat_history.append({"role": "assistant", "content": response})