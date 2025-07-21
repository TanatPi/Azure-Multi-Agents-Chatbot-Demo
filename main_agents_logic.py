import asyncio
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from promptflow_logics.news_agents_logic import get_news_agent_response
from promptflow_logics.callcenter_agents_logic import get_callcenter_agent_response
from semantic_kernel.contents.utils.author_role import AuthorRole
import streamlit as st

#### === Helper functions ===
# === Helper to count token ===
import tiktoken

tokenizer_4o = tiktoken.encoding_for_model("gpt-4o-mini") 
tokenizer_4_1 = tiktoken.get_encoding("o200k_base") # gpt-4.1

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


# === Final Agent Flows ===
async def get_agent_response(user_query: str, chat_history, main_thread, user_thread, agents):
    # agents
    router_agent = agents["router_agent"]
    reply_agent = agents["reply_agent"]
    pdf_rag_agent = agents["pdf_rag_agent"]
    callcenter_rag_agent = agents["callcenter_rag_agent"]
    keyword_extractor_agent = agents["keyword_extractor_agent"]
    news_orchestrator_agent = agents["news_orchestrator_agent"]
    # custom search tools
    pdf_search = agents["pdf_search"]
    callcenter_search = agents["callcenter_search"]


    input_tokens_router = None
    output_tokens_router = None
    rag_prompt_tokens = None
    rag_completion_tokens = None
    input_tokens_orchestrator = None
    output_tokens_orchestrator = None
    keyword_input_tokens = None
    keyword_output_tokens = None
    input_tokens_reply = None
    output_tokens_reply = None

    # === ROUTER ===
    router_user_message = ChatMessageContent(role=AuthorRole.USER, content=user_query)

    ### Run router
    async for route in router_agent.invoke(messages=[router_user_message], thread=user_thread):
        route_str = str(route).strip()
        user_thread = route.thread

    # Parse router output
    intent = None
    language = None
    for line in route_str.splitlines():
        if line.startswith("INTENT:"):
            intent = line.split("INTENT:")[1].strip()
        elif line.startswith("LANGUAGE:"):
            language = line.split("LANGUAGE:")[1].strip()

    # Fallback in case parsing fails
    intent = intent or "BYPASS"
    language = language or "THAI"
    
    # === Token count for router input
    input_tokens_router = count_tokens(user_query,tokenizer_4_1)
    # === Token count for router output
    output_tokens_router = count_tokens(route_str,tokenizer_4_1)

    if intent == "NEWS":
        # === Create step placeholders ===
        keyword_status = st.empty()
        rag_status = st.empty()
        orch_status = st.empty()
        status = {
                "keyword": keyword_status,
                "rag": rag_status,
                "orchestrator": orch_status,
                }
        # Run news agents flow in sync context
        final_response, main_thread, rag_prompt_tokens, rag_completion_tokens, input_tokens_orchestrator, output_tokens_orchestrator, keyword_input_tokens, keyword_output_tokens   = await get_news_agent_response(
                user_query,
                user_thread,
                news_orchestrator_agent,
                pdf_rag_agent,
                keyword_extractor_agent,
                pdf_search,
                language,
                status
            )
        status["orchestrator"].empty()

    elif intent == "CALLCENTER":
        # === Create step placeholders ===
        keyword_status = st.empty()
        rag_status = st.empty()
        orch_status = st.empty()
        status = {
                "keyword": keyword_status,
                "rag": rag_status
                }
        # Run news agents flow in sync context
        final_response, main_thread, rag_prompt_tokens, rag_completion_tokens, keyword_input_tokens, keyword_output_tokens = await get_callcenter_agent_response(
                user_query,
                user_thread,
                callcenter_rag_agent,
                keyword_extractor_agent,
                callcenter_search,
                language,
                status
            )
        status["rag"].empty()

    else:
        reply_user_prompt = f"Since other agent are bypassed, take the chat history and answer {user_query} in {language} accordingly if possible."
        reply_user_message = ChatMessageContent(role=AuthorRole.USER, content=reply_user_prompt)
            
        # === Token count for reply agent input 
        input_tokens_reply = count_tokens(reply_user_prompt,tokenizer_4_1)

        final_response = ""
        async for reply in reply_agent.invoke(messages=[reply_user_message],thread=main_thread):
            final_response = str(reply)
            main_thread = reply.thread
    
        # === Token count for reply agent output
        output_tokens_reply = count_tokens(final_response,tokenizer_4_1)

    chat_history.add_user_message(user_query)
    chat_history.add_assistant_message(final_response)

    return (
        final_response,
        chat_history,
        main_thread,
        user_thread,
        rag_prompt_tokens,
        rag_completion_tokens,
        input_tokens_orchestrator,
        output_tokens_orchestrator,
        keyword_input_tokens,
        keyword_output_tokens,
        input_tokens_reply,
        output_tokens_reply,
        input_tokens_router,
        output_tokens_router
    )