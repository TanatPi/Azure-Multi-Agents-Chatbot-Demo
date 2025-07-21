import asyncio
from semantic_kernel.contents.chat_message_content import ChatMessageContent

#### === Helper functions ===
# === Helper to count token ===
import tiktoken

tokenizer_rag = tiktoken.encoding_for_model("gpt-4o-mini") 
tokenizer_orchestrator = tiktoken.get_encoding("o200k_base") # gpt-4.1

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

# === Helper to call one sub-agent ===
async def run_mmrag_agent(agents, search, user_query, search_keywords, filter=None, top_k=10):
    context_text, context_table, context_image = await asyncio.gather(
        search.search_text_content(search_keywords, filter=filter, top_k=top_k),
        search.search_table_content(search_keywords, filter=filter, top_k=top_k),
        search.search_image_content(search_keywords, filter=filter, top_k=top_k),
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

    input_tokens = count_tokens(user_prompt,tokenizer_rag)

    user_message = ChatMessageContent(role="user", content=user_prompt)

    response_text = ""
    async for response in agents.invoke(messages=[user_message]):
        response_text = str(response)

    output_tokens = count_tokens(response_text,tokenizer_rag)

    return response_text, input_tokens, output_tokens


# === Final Agent Flows ===
async def get_agent_response(user_query: str, thread, orchestrator_agent, pdf_rag_agent, keyword_extractor_agent, pdf_search) -> tuple[str, str, int, int, int]:

    ##### 1. Run Keyword extractor
    keyword_agent_user_prompt = f"Extract keywords from this query: {user_query}"
    keyword_agent_message = ChatMessageContent(role="user", content=keyword_agent_user_prompt)# === Token count for input to keyword agent
    
    # === Token count for input to keyword agent
    keyword_input_tokens = count_tokens(keyword_agent_user_prompt, tokenizer_orchestrator)

    search_keywords = user_query
    async for response in keyword_extractor_agent.invoke(messages=[keyword_agent_message]):
        search_keywords = str(response)

    # === Token count for keyword agent response
    keyword_output_tokens = count_tokens(search_keywords, tokenizer_orchestrator)

    ##### 2. Run Search agents concurrently
    # Run all sub-agents concurrently and collect token counts
    results = await asyncio.gather(
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords,filter="key_prefix eq 'monthlystandpoint'", top_k=4),
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords,filter="key_prefix eq 'ktm'", top_k=7),
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords,filter="key_prefix eq 'kcma'", top_k=12),
    )

    responses = [r[0] for r in results]

    # === Token count for search agents
    rag_prompt_tokens = sum(r[1] for r in results)
    rag_completion_tokens = sum(r[2] for r in results)

    ##### 3. Run Orchestrator
    orchestrator_prompt = f"""You are the final assistant. Your job is to synthesize and consolidate the following three answers into a single, coherent, complete response for the following question:

        Original question:
        {user_query}

        Answer from monthlystandpoint:
        {responses[0]}

        Answer from KTM:
        {responses[1]}

        Answer from KCMA:
        {responses[2]}

        Please write your final response in a clear, structured way. Make sure no important point is missed.
        """

    orchestrator_message = ChatMessageContent(role="user", content=orchestrator_prompt)

    # === Token count for input to orchestrator
    input_tokens_orchestrator = count_tokens(orchestrator_prompt,tokenizer_orchestrator)

    final_response = ""
    async for orchestration in orchestrator_agent.invoke(messages=[orchestrator_message], thread=thread):
        final_response = str(orchestration)
        thread = orchestration.thread

    # === Token count for output to orchestrator
    output_tokens_orchestrator = count_tokens(final_response,tokenizer_orchestrator)

    return (
        final_response,
        thread,
        rag_prompt_tokens,
        rag_completion_tokens,
        input_tokens_orchestrator,
        output_tokens_orchestrator,
        keyword_input_tokens,
        keyword_output_tokens,
    )