import streamlit as st
import asyncio

from agents_logic import get_agent_response

st.set_page_config(page_title="Economic News GPT Chatbot", page_icon="💬", layout="wide")
st.title("💬 Economic News Chatbot")

if "thread" not in st.session_state:
    st.session_state.thread = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display full history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask me anything about KAsset reports...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_response, thread = asyncio.run(
                get_agent_response(user_query, thread=st.session_state.thread)
            )
            st.session_state.thread = thread
            st.markdown(assistant_response)

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
