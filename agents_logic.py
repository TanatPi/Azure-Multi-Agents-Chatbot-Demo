import asyncio
import streamlit as st
from semantic_kernel.contents.chat_message_content import ChatMessageContent

from agents.mm_rag_agent import (
    get_mm_rag_agent as get_agent,
    get_search_plugin as get_search,
)


# === Initialize the 3 sub-agents ===
pdf_rag_agent = get_agent()
pdf_search = get_search(
    text_index_name="pdf-economic-summary",
    table_index_name="pdf-economic-summary-tables",
    image_index_name="pdf-economic-summary-images"
)

# === Initialize the final orchestrator agent (same model, but could be a different prompt/setup) ===
from agents.ochestrator_agent import get_ochestrator_agent
ochestrator_agent = asyncio.run(get_ochestrator_agent())


# === Helper to call one sub-agent ===
async def run_agent(agent, search, user_query, filter=None):
    context_text, context_table, context_image = await asyncio.gather(
        search.search_text_content(user_query, filter=filter),
        search.search_table_content(user_query, filter=filter),
        search.search_image_content(user_query, filter=filter),
    )

    user_prompt = f"""Use the following JSON context to answer the question:

        Context text data:
        {context_text}

        Context table data:
        {context_table}

        Context image data:
        {context_image}

        Question: {user_query}
        """

    user_message = ChatMessageContent(role="user", content=user_prompt)

    response_text = ""
    async for response in agent.invoke(messages=[user_message]):
        response_text = str(response)

    return response_text


# === Final Orchestrator ===
async def get_agent_response(user_query: str, thread=None) -> tuple[str, str, str, str]:
    # Step 1: Run the 3 sub-agents in parallel
    response_1, response_2, response_3 = await asyncio.gather(
        run_agent(pdf_rag_agent, pdf_search, user_query, filter="key_prefix eq 'monthlystandpoint'"),
        run_agent(pdf_rag_agent, pdf_search, user_query, filter="key_prefix eq 'ktm'"),
        run_agent(pdf_rag_agent, pdf_search, user_query, filter="key_prefix eq 'kcma'"),
    )

    # Step 2: Combine them and send to the ochestrator agent
    ochestrator_prompt = f"""You are the final assistant. Your job is to synthesize and consolidate the following three answers into a single, coherent, complete response for the user:

        Answer from monthlystandpoint:
        {response_1}

        Answer from KTM:
        {response_2}

        Answer from KCMA:
        {response_3}

        Please write your final response in a clear, structured way. Make sure no important point is missed.
        """

    ochestrator_message = ChatMessageContent(role="user", content=ochestrator_prompt)

    final_response = ""
    async for ochestration in ochestrator_agent.invoke(messages=[ochestrator_message], thread=thread):
        final_response = str(ochestration)
        thread = ochestration.thread  # update thread for memory

    return final_response, thread
