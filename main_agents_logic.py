import asyncio
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from promptflow_logics.news_agents_logic import get_news_agent_response
from promptflow_logics.callcenter_agents_logic import get_callcenter_agent_response
from promptflow_logics.fundfact_agents_logic import get_fundfact_agent_response
from semantic_kernel.contents.utils.author_role import AuthorRole
import streamlit as st

# === Tokenizer setup ===
import tiktoken

tokenizer_4o = tiktoken.encoding_for_model("gpt-4o-mini")
tokenizer_4_1 = tiktoken.get_encoding("o200k_base")  # gpt-4.1

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


async def get_agent_response(user_query: str, chat_history: ChatHistoryAgentThread, main_thread, user_thread, agents, container):
    has_streamed = False  # Flag for streaming output

    # Unpack agents and tools
    main_router_agent = agents["main_router_agent"]
    news_router_agent = agents["news_router_agent"]
    reply_agent = agents["reply_agent"]
    pdf_rag_agent = agents["pdf_rag_agent"]
    callcenter_rag_agent = agents["callcenter_rag_agent"]
    keyword_extractor_agent = agents["keyword_extractor_agent"]
    news_orchestrator_agent = agents["news_orchestrator_agent"]
    fundfact_linguistic_rag_agent = agents["fundfact_linguistic_rag_agent"]
    fundfact_coder_rag_agent = agents["fundfact_coder_rag_agent"]
    fundfact_orchestrator_agent = agents["fundfact_orchestrator_agent"]

    # Search plugins
    pdf_search = agents["pdf_search"]
    callcenter_search = agents["callcenter_search"]
    fundfact_linguistic_search = agents["fundfact_linguistic_search"]

    # Initialize token counts to None
    input_tokens_router = output_tokens_router = None
    rag_prompt_tokens = rag_completion_tokens = None
    input_tokens_orchestrator = output_tokens_orchestrator = None
    keyword_input_tokens = keyword_output_tokens = None
    input_tokens_reply = output_tokens_reply = None

    # === Run main router ===
    router_user_message = ChatMessageContent(role=AuthorRole.USER, content=user_query)

    async for route in main_router_agent.invoke(messages=[router_user_message], thread=user_thread):
        route_str = str(route).strip()
        user_thread = route.thread

    # Parse intent and language from router output
    intent = None
    language = None
    for line in route_str.splitlines():
        if line.startswith("INTENT:"):
            intent = line.split("INTENT:")[1].strip()
        elif line.startswith("LANGUAGE:"):
            language = line.split("LANGUAGE:")[1].strip()

    # Defaults if parsing fails
    intent = intent or "BYPASS"
    language = language or "THAI"

    # Token counts for router
    input_tokens_router = count_tokens(user_query, tokenizer_4_1)
    output_tokens_router = count_tokens(route_str, tokenizer_4_1)

    if intent == "NEWS":
        # Streamlit UI placeholders for feedback
        routing_status = st.empty()
        keyword_status = st.empty()
        rag_status = st.empty()
        orch_status = st.empty()
        status = {
            "router": routing_status,
            "keyword": keyword_status,
            "rag": rag_status,
            "orchestrator": orch_status,
        }

        # Run news flow
        (
            final_response, thread,
            sub_input_tokens_router, sub_output_tokens_router,
            rag_prompt_tokens, rag_completion_tokens,
            input_tokens_orchestrator, output_tokens_orchestrator,
            keyword_input_tokens, keyword_output_tokens,
            has_streamed
        ) = await get_news_agent_response(
            user_query, user_thread, main_thread,
            news_router_agent, news_orchestrator_agent,
            pdf_rag_agent, keyword_extractor_agent,
            pdf_search, language, status, container
        )
        input_tokens_router += sub_input_tokens_router
        output_tokens_orchestrator += sub_output_tokens_router
        status["orchestrator"].empty()

        # Merge thread histories if needed
        if main_thread is not None and thread is not main_thread:
            for msg in thread._chat_history:
                main_thread._chat_history.add_message(msg)
        else:
            main_thread = thread

    elif intent == "CALLCENTER":
        keyword_status = st.empty()
        rag_status = st.empty()
        status = {"keyword": keyword_status, "rag": rag_status}

        # Run call center flow
        (
            final_response, thread,
            rag_prompt_tokens, rag_completion_tokens,
            keyword_input_tokens, keyword_output_tokens,
            has_streamed
        ) = await get_callcenter_agent_response(
            user_query, user_thread,
            callcenter_rag_agent,
            keyword_extractor_agent,
            callcenter_search,
            language, status, container
        )
        status["rag"].empty()

        if main_thread is not None and thread is not main_thread:
            for msg in thread._chat_history:
                main_thread._chat_history.add_message(msg)
        else:
            main_thread = thread

    elif intent == "FUNDFACT":
        keyword_status = st.empty()
        rag_status = st.empty()
        orch_status = st.empty()
        status = {
            "keyword": keyword_status,
            "rag": rag_status,
            "orchestrator": orch_status,
        }

        # Run fund fact flow
        (
            final_response, thread,
            rag_prompt_tokens, rag_completion_tokens,
            input_tokens_orchestrator, output_tokens_orchestrator,
            keyword_input_tokens, keyword_output_tokens,
            has_streamed
        ) = await get_fundfact_agent_response(
            user_query, user_thread,
            keyword_extractor_agent,
            fundfact_linguistic_rag_agent,
            fundfact_linguistic_search,
            fundfact_coder_rag_agent,
            fundfact_orchestrator_agent,
            language, status, container
        )
        status["orchestrator"].empty()

        if main_thread is not None and thread is not main_thread:
            for msg in thread._chat_history:
                main_thread._chat_history.add_message(msg)
        else:
            main_thread = thread

    else:
        # Fallback to generic reply agent
        reply_user_prompt = f"Since other agents are bypassed, take the chat history and answer {user_query} in {language} accordingly if possible."
        reply_user_message = ChatMessageContent(role=AuthorRole.USER, content=reply_user_prompt)

        input_tokens_reply = count_tokens(reply_user_prompt, tokenizer_4_1)

        final_response = ""
        async for reply in reply_agent.invoke_stream(messages=[reply_user_message], thread=main_thread):
            final_response += str(reply)
            container.markdown(final_response)
            main_thread = reply.thread
            has_streamed = True

        output_tokens_reply = count_tokens(final_response, tokenizer_4_1)

    # Add messages to chat history
    chat_history.add_user_message(user_query)
    chat_history.add_assistant_message(final_response)

    # Return all useful info for metrics and state
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
        output_tokens_router,
        has_streamed
    )
