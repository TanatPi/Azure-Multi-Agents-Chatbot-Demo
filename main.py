import streamlit as st
import asyncio
import time 

# not need in real st deployment
from dotenv import load_dotenv
load_dotenv()

from main_agents_logic import (get_agent_response)
from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
# === Initialize Agents ===
from agents.reply_agent import get_reply_agent
from agents.router_agent import get_router_agent
from agents.mm_rag_agent import (
    get_mm_rag_agent,
    get_mm_search_plugin,
)
from agents.txt_rag_agent import (
    get_txt_rag_agent,
    get_txt_search_plugin,
)
from agents.orchestrator_agent import get_orchestrator_agent
from agents.keyword_extractor_agent import get_keyword_extractor_agent


# === Initialize Streamlit UI ===
st.set_page_config(page_title="JOHN-AI Chatbot", page_icon="💬", layout="wide")
st.title("💬 JOHN-AI Chatbot")

# === Initialize session state for agent and memory ===
if "thread" not in st.session_state:
    st.session_state.thread = None
if "user_thread" not in st.session_state:
    st.session_state.user_thread = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    # Add welcome message only once (when chat_history is first created)
    with st.chat_message("assistant"):
        welcome_message = "👋 สวัสดีครับ! ผมชื่อจอห์น หากคุณมีคำถามเกี่ยวกับข่าวสารเศรษฐกิจ การลงทุน หรือ อื่น ๆ สามารถสอบถามได้เลยครับ!"
        st.markdown(welcome_message)
# agents
if "router_agent" not in st.session_state:
    st.session_state.router_agent = None
if "reply_agent" not in st.session_state:
    st.session_state.reply_agent = None
if "news_orchestrator_agent" not in st.session_state:
    st.session_state.news_orchestrator_agent = None
if "pdf_rag_agent" not in st.session_state:
    st.session_state.pdf_rag_agent = None
if "callcenter_rag_agent" not in st.session_state:
    st.session_state.pdf_rag_agent = None
if "keyword_extractor_agent" not in st.session_state:
    st.session_state.keyword_extractor_agent = None
if "pdf_search" not in st.session_state:
    st.session_state.pdf_search = None
if "callcenter_search" not in st.session_state:
    st.session_state.callcenter_search = None

# === Shared Kernel Initialization ===
def initialize_kernel():
    kernel = Kernel()

    return kernel

# === Async initializer ===
async def initialize_agents():
    # Create or get shared kernel
    if "shared_kernel" not in st.session_state:
        st.session_state.shared_kernel = initialize_kernel()

    kernel = st.session_state.shared_kernel

    agents = {
        "router_agent": get_router_agent(kernel),
        "callcenter_search": get_txt_search_plugin(
            text_index_name="callcenterinfo",
        ),
        "pdf_search": get_mm_search_plugin(
            text_index_name="pdf-economic-summary",
            table_index_name="pdf-economic-summary-tables",
            image_index_name="pdf-economic-summary-images"
        ),
        "pdf_rag_agent": get_mm_rag_agent(kernel),
        "callcenter_rag_agent": get_txt_rag_agent(kernel, "callcenter_rag_agent"),
        "reply_agent": get_reply_agent(kernel),
        "keyword_extractor_agent": get_keyword_extractor_agent(kernel),
        "news_orchestrator_agent": get_orchestrator_agent(kernel,"news_orchestrator"),
    }
    st.session_state.agents = agents
    st.session_state.initialized = True

# === Force sync for async init at top ===
if not st.session_state.initialized:
    asyncio.run(initialize_agents())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()
    st.session_state.chat_history.add_assistant_message(welcome_message)
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

# === Proper async function to read async iterable from thread
async def display_thread_messages(thread):
    st.markdown("### 🧵 Thread Info")
    st.markdown(f"- **Thread ID**: `{thread.id}`")

    st.markdown("#### 📜 Thread Messages")
    i = 1
    async for msg in thread.get_messages():
        if isinstance(msg, ChatMessageContent):
            st.markdown(f"**{i}. {msg.role}**: {msg.content}")
            i += 1

# === Chat input ===
user_query = st.chat_input("Ask me anything about economic reports...")

# === If user submits a query ===
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            # Run async logic in sync context
            (response, chat_history, thread, user_thread, rag_prompt, rag_completion, 
             orch_prompt, orch_completion, keyword_prompt, keyword_completion, input_tokens_reply, output_tokens_reply,
                      input_tokens_router, output_tokens_router )= asyncio.run(
                get_agent_response(
                    user_query,
                    st.session_state.chat_history,
                    st.session_state.thread,
                    st.session_state.user_thread,
                    st.session_state.agents,
                )
            )

            end_time = time.time()
            total_time = end_time - start_time

            st.markdown(response)
            st.session_state.thread = thread
            st.session_state.user_thread = user_thread
            st.session_state.chat_history = chat_history
            # === Debug: Print thread info
            # if st.session_state.thread:
            #     asyncio.run(display_thread_messages(st.session_state.thread))
            st.markdown(f"⏱️ *Response generated in {total_time:.2f} seconds*")

            # === Print only if each token count exists ===
            if input_tokens_router is not None and output_tokens_router is not None:
                st.markdown(
                    f"📊 *Router tokens: {input_tokens_router} prompt + {output_tokens_router} completion = {input_tokens_router+ output_tokens_router} total*"
                )
            if keyword_prompt is not None and keyword_completion is not None:
                st.markdown(
                    f"📊 *Keyword Extractor tokens: {keyword_prompt} prompt + {keyword_completion} completion = {keyword_prompt + keyword_completion} total*"
                )

            if rag_prompt is not None and rag_completion is not None:
                st.markdown(
                    f"📊 *RAG agents tokens: {rag_prompt} prompt + {rag_completion} completion = {rag_prompt + rag_completion} total*"
                )

            if orch_prompt is not None and orch_completion is not None:
                st.markdown(
                    f"📊 *Orchestrator tokens: {orch_prompt} prompt + {orch_completion} completion = {orch_prompt + orch_completion} total*"
                )
            if input_tokens_reply is not None and output_tokens_reply is not None:
                st.markdown(
                    f"📊 *Reply agent tokens: {input_tokens_reply} prompt + {output_tokens_reply} completion = {input_tokens_reply + output_tokens_reply} total*"
                )