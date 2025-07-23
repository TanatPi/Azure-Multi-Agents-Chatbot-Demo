import asyncio
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole



#### === Helper functions ===
# === Helper to count token ===
import tiktoken

tokenizer_4o = tiktoken.encoding_for_model("gpt-4o-mini") 
tokenizer_4_1 = tiktoken.get_encoding("o200k_base") # gpt-4.1

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

# === Helper to call one sub-agent ===
async def run_mmrag_agent(agents, search, user_query, search_keywords, filter=None, top_k=10):
    context_text, context_table, context_image = await asyncio.gather(
        search.search_text_content(search_keywords, filter=filter, top_k=top_k),
        search.search_table_content(search_keywords, filter=filter, top_k=5),
        search.search_image_content(search_keywords, filter=filter, top_k=4),
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

    input_tokens = count_tokens(user_prompt,tokenizer_4o)

    user_message = ChatMessageContent(role=AuthorRole.USER, content=user_prompt)

    response_text = ""
    async for response in agents.invoke(messages=[user_message]):
        response_text = str(response)

    output_tokens = count_tokens(response_text,tokenizer_4o)

    return response_text, input_tokens, output_tokens


# === Final Agent Flows ===
async def get_news_agent_response(user_query: str, user_thread,orchestrator_agent, pdf_rag_agent, keyword_extractor_agent, pdf_search,language, status,container):
    ##### 1. Run Keyword extractor
    if status: status["keyword"].markdown("🔍 Extracting keywords...")
    keyword_agent_user_prompt = f"Extract keywords from this query: {user_query}"
    keyword_agent_message = ChatMessageContent(role=AuthorRole.USER, content=keyword_agent_user_prompt)# === Token count for input to keyword agent
    
    # === Token count for input to keyword agent
    keyword_input_tokens = count_tokens(keyword_agent_user_prompt, tokenizer_4_1)

    search_keywords = user_query
    async for response in keyword_extractor_agent.invoke(messages=[keyword_agent_message], thread = user_thread):
        search_keywords = str(response)

    # === Token count for keyword agent response
    keyword_output_tokens = count_tokens(search_keywords, tokenizer_4_1)

    ##### 2. Run Search agents concurrently
    if status: 
        status["rag"].markdown("📚 Running RAG agents...")
        status["keyword"].empty()
    # Run all sub-agents concurrently and collect token counts
    results = await asyncio.gather(
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords,filter="key_prefix eq 'monthlystandpoint'", top_k=10),
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords,filter="key_prefix eq 'ktm'", top_k=20),
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords,filter="key_prefix eq 'kcma'", top_k=40),
    )

    responses = [r[0] for r in results]

    # === Token count for search agents
    rag_prompt_tokens = sum(r[1] for r in results)
    rag_completion_tokens = sum(r[2] for r in results)

    ##### 3. Run Orchestrator
    if status: 
        status["orchestrator"].markdown("🧠 Synthesizing final RAG response...")
        status["rag"].empty()
    orchestrator_prompt = f"""The information given to you are:

        information from Monthly Standpoint (monthlystandpoint) document:
        {responses[0]}

        information from Know the Markets (KTM) document:
        {responses[1]}

        information from KAsset Capital Market Assumptions (KCMA) document:
        {responses[2]}

        If {user_query} mention specific documents, left out what is not stated.
        Otherwise, consider all three, but prioritize KCMA and monthlystandpoint over KTM in terms of correctness.
        Cross-check the fact and use those to answer the original question:
        {user_query}

        Please write your final response in a clear {language}, structured way. Make sure no important point is missed.
        """

    orchestrator_message = ChatMessageContent(role=AuthorRole.USER, content=orchestrator_prompt)

    # === Token count for input to orchestrator
    input_tokens_orchestrator = count_tokens(orchestrator_prompt,tokenizer_4_1)

    final_response = ""
    async for orchestration in orchestrator_agent.invoke_stream(messages=[orchestrator_message]):
        final_response += str(orchestration)
        container.markdown(final_response)
        main_thread = orchestration.thread
        has_streamed = True

    # === Token count for output to orchestrator
    output_tokens_orchestrator = count_tokens(final_response,tokenizer_4_1)

    return (
        final_response,
        main_thread,
        rag_prompt_tokens,
        rag_completion_tokens,
        input_tokens_orchestrator,
        output_tokens_orchestrator,
        keyword_input_tokens,
        keyword_output_tokens,
        has_streamed
    )