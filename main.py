import streamlit as st
import asyncio
import time

# For local dev only, not needed in production deployment
from dotenv import load_dotenv
load_dotenv()

from main_agents_logic import get_agent_response
from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.agents import ChatHistoryAgentThread

# Agents imports
from agents.reply_agent import get_reply_agent
from agents.router_agent import get_router_agent
from agents.mm_rag_agent import get_mm_rag_agent, get_mm_search_plugin
from agents.txt_rag_agent import get_txt_rag_agent, get_txt_search_plugin
from agents.orchestrator_agent import get_orchestrator_agent
from agents.keyword_extractor_agent import get_keyword_extractor_agent
from agents.fundfact_coder_rag_agent import get_fundfact_coder_rag_agent

# === Streamlit UI setup ===
st.set_page_config(page_title="WIN-AI Chatbot", page_icon="💬", layout="wide")
st.title("💬 WIN-AI Chatbot")

# Initialize session state for chat threads and agents
if "thread" not in st.session_state:
    st.session_state.thread = None
if "user_thread" not in st.session_state:
    st.session_state.user_thread = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Welcome message shown once at start
welcome_message = (
    "👋 สวัสดีครับ! ผมชื่อวินัย (WIN-AI) หากคุณมีคำถามเกี่ยวกับข่าวสารเศรษฐกิจ "
    "การลงทุน หรือ KAsset สามารถสอบถามได้เลยครับ!"
)
if not st.session_state.initialized:
    with st.chat_message("assistant"):
        st.markdown(welcome_message)

# Setup agent placeholders in session state
agent_keys = [
    "main_router_agent", "news_router_agent", "reply_agent", "news_orchestrator_agent",
    "fundfact_orchestrator_agent", "fundfact_coder_rag_agent", "fundfact_linguistic_rag_agent",
    "pdf_rag_agent", "callcenter_rag_agent", "keyword_extractor_agent",
    "pdf_search", "callcenter_search", "fundfact_linguistic_search"
]
for key in agent_keys:
    if key not in st.session_state:
        st.session_state[key] = None

# Kernel and agents initialization
def initialize_kernel():
    return Kernel()

async def initialize_agents():
    if "shared_kernel" not in st.session_state:
        st.session_state.shared_kernel = initialize_kernel()

    kernel = st.session_state.shared_kernel

    agents = {
        "main_router_agent": get_router_agent(kernel, "main_router_agent"),
        "news_router_agent": get_router_agent(kernel, "news_router_agent"),
        "fundfact_linguistic_search": get_txt_search_plugin(text_index_name="mutualfunds"),
        "callcenter_search": get_txt_search_plugin(text_index_name="callcenterinfo"),
        "pdf_search": get_mm_search_plugin(
            text_index_name="pdf-economic-summary",
            table_index_name="pdf-economic-summary-tables",
            image_index_name="pdf-economic-summary-images",
        ),
        "pdf_rag_agent": get_mm_rag_agent(kernel),
        "callcenter_rag_agent": get_txt_rag_agent(kernel, "callcenter_rag_agent"),
        "reply_agent": get_reply_agent(kernel),
        "keyword_extractor_agent": get_keyword_extractor_agent(kernel),
        "news_orchestrator_agent": get_orchestrator_agent(kernel, "news_orchestrator"),
        "fundfact_orchestrator_agent": get_orchestrator_agent(kernel, "fundfact_orchestrator"),
        "fundfact_linguistic_rag_agent": get_txt_rag_agent(kernel, "fundfact_linguistic_rag_agent"),
        "fundfact_coder_rag_agent": await get_fundfact_coder_rag_agent(),
    }
    st.session_state.agents = agents
    st.session_state.thread = None
    st.session_state.initialized = True

# Initialize agents synchronously in Streamlit start
if not st.session_state.initialized:
    asyncio.run(initialize_agents())

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()
    st.session_state.chat_history.add_assistant_message(welcome_message)
else:
    # Replay chat history messages on UI
    for msg in st.session_state.chat_history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

# Chat input box
user_query = st.chat_input("Ask me anything about economic reports...")

if user_query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            streamed_output_container = st.empty()

            (
                response,
                chat_history,
                thread,
                user_thread,
                rag_prompt,
                rag_completion,
                orch_prompt,
                orch_completion,
                keyword_prompt,
                keyword_completion,
                input_tokens_reply,
                output_tokens_reply,
                input_tokens_router,
                output_tokens_router,
                has_streamed,
            ) = asyncio.run(
                get_agent_response(
                    user_query,
                    st.session_state.chat_history,
                    st.session_state.thread,
                    st.session_state.user_thread,
                    st.session_state.agents,
                    streamed_output_container,
                )
            )

            end_time = time.time()
            total_time = end_time - start_time

            # Show full response if not streamed
            if not has_streamed:
                st.markdown(response)

            # Update session states
            st.session_state.thread = thread
            st.session_state.user_thread = user_thread
            st.session_state.chat_history = chat_history

            st.markdown(f"⏱️ *Response generated in {total_time:.2f} seconds*")

            # Display token usage breakdown if available
            if input_tokens_router is not None and output_tokens_router is not None:
                st.markdown(f"📊 *Router tokens: {input_tokens_router} prompt + {output_tokens_router} completion = {input_tokens_router + output_tokens_router} total*")
            if keyword_prompt is not None and keyword_completion is not None:
                st.markdown(f"📊 *Keyword Extractor tokens: {keyword_prompt} prompt + {keyword_completion} completion = {keyword_prompt + keyword_completion} total*")
            if rag_prompt is not None and rag_completion is not None:
                st.markdown(f"📊 *RAG agents tokens: {rag_prompt} prompt + {rag_completion} completion = {rag_prompt + rag_completion} total*")
            if orch_prompt is not None and orch_completion is not None:
                st.markdown(f"📊 *Orchestrator tokens: {orch_prompt} prompt + {orch_completion} completion = {orch_prompt + orch_completion} total*")
            if input_tokens_reply is not None and output_tokens_reply is not None:
                st.markdown(f"📊 *Reply agent tokens: {input_tokens_reply} prompt + {output_tokens_reply} completion = {input_tokens_reply + output_tokens_reply} total*")
