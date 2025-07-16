import asyncio
from semantic_kernel.contents.chat_message_content import ChatMessageContent

# === Helper to call one sub-agent ===
async def run_mmrag_agent(agent, search, user_query, filter=None, top_k=10):
    context_text, context_table, context_image = await asyncio.gather(
        search.search_text_content(user_query, filter=filter, top_k= top_k),
        search.search_table_content(user_query, filter=filter, top_k= top_k),
        search.search_image_content(user_query, filter=filter, top_k= top_k),
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
async def get_agent_response(user_query: str, thread, ochestrator_agent, pdf_rag_agent, pdf_search) -> tuple[str, str]:
    response_1, response_2, response_3 = await asyncio.gather(
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, filter="key_prefix eq 'monthlystandpoint'", top_k = 5),
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, filter="key_prefix eq 'ktm'", top_k = 10),
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, filter="key_prefix eq 'kcma'", top_k = 20),
    )

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
        thread = ochestration.thread

    return final_response, thread
