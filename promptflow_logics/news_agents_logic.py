import asyncio
import ast
import math
from datetime import datetime

from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

import tiktoken

# === Constants ===
today_str = datetime.now().strftime("%B %d, %Y")  # e.g., "July 24, 2025"

# === Tokenizers for counting tokens with different models ===
tokenizer_4o = tiktoken.encoding_for_model("gpt-4o-mini")
tokenizer_4_1 = tiktoken.get_encoding("o200k_base")  # gpt-4.1

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a text string given a tokenizer."""
    return len(tokenizer.encode(text))


# === Helper to call one RAG sub-agent with text and table search ===
async def run_mmrag_agent(agents, search, user_query, search_keywords, filter=None, top_k=10):
    # Concurrently search text and tables for context
    context_text, context_table = await asyncio.gather(
        search.search_text_content(search_keywords, filter=filter, top_k=top_k),
        search.search_table_content(search_keywords, filter=filter, top_k=5),
    )

    user_prompt = f"""Use the following JSON context to answer the question:

        Context text data:
        {context_text}

        Context table data:
        {context_table}

        Question: {user_query}
        """

    input_tokens = count_tokens(user_prompt, tokenizer_4o)
    user_message = ChatMessageContent(role=AuthorRole.USER, content=user_prompt)

    response_text = ""
    async for response in agents.invoke(messages=[user_message]):
        response_text = str(response)

    output_tokens = count_tokens(response_text, tokenizer_4o)
    return response_text, input_tokens, output_tokens


# === Main flow for getting news agent response ===
async def get_news_agent_response(
    user_query: str,
    user_thread,
    main_thread,
    news_router_agent,
    orchestrator_agent,
    pdf_rag_agent,
    keyword_extractor_agent,
    pdf_search,
    language,
    status,
    container
):
    # Step 0: Run the router agent to get route scores
    if status:
        status["router"].markdown("🔀 Selecting appropriate documents...")

    router_prompt_with_date = f"Today is {today_str}.\n{user_query}"
    router_user_message = ChatMessageContent(role=AuthorRole.USER, content=router_prompt_with_date)
    async for route in news_router_agent.invoke(messages=[router_user_message]):
        route_str = str(route).strip()
        user_thread = route.thread

    # Token counts for router input and output
    input_tokens_router = count_tokens(user_query, tokenizer_4_1)
    output_tokens_router = count_tokens(route_str, tokenizer_4_1)

    # Parse router output (expect a dict with scores)
    try:
        route_scores = ast.literal_eval(route_str)
        assert isinstance(route_scores, dict)
        valid_routes = {"MONTHLYSTANDPOINT", "KCMA", "KTM"}
        route_scores = {k: v for k, v in route_scores.items() if k in valid_routes and isinstance(v, (int, float))}
    except Exception:
        # Fallback if parsing fails
        route_scores = {"MONTHLYSTANDPOINT": 10, "KCMA": 0, "KTM": 0}

    # Step 1: Keyword extraction
    if status:
        status["keyword"].markdown("🔍 Extracting keywords...")
        status["router"].empty()

    keyword_agent_user_prompt = f"Extract keywords from this query: {user_query}"
    keyword_agent_message = ChatMessageContent(role=AuthorRole.USER, content=keyword_agent_user_prompt)
    keyword_input_tokens = count_tokens(keyword_agent_user_prompt, tokenizer_4_1)

    search_keywords = user_query
    async for response in keyword_extractor_agent.invoke(messages=[keyword_agent_message], thread=user_thread):
        search_keywords = str(response)

    keyword_output_tokens = count_tokens(search_keywords, tokenizer_4_1)

    # Step 2: Run RAG agents with score-based adjusted top_k
    if status:
        status["rag"].markdown("📚 Running RAG agents...")
        status["keyword"].empty()

    # Base top_k per route
    default_top_k = {
        "MONTHLYSTANDPOINT": 20,
        "KTM": 15,
        "KCMA": 35,
    }

    # Nonlinear bias for route scores
    def biased_score(route, score):
        if route == "MONTHLYSTANDPOINT":
            # sqrt boost for MONTHLYSTANDPOINT
            return min(math.sqrt(score) * math.sqrt(10), 10)
        elif route == "KTM":
            # squared penalty for KTM
            return min((score ** 2) / 10, 10)
        elif route == "KCMA":
            # linear bounded score for KCMA
            return min(score, 10)
        else:
            return score

    # Calculate adjusted top_k for each route
    adjusted_top_k = {
        route: max(1, int(default_top_k[route] * (biased_score(route, score) / 10)))
        for route, score in route_scores.items() if score > 0
    }

    route_filters = {
        "MONTHLYSTANDPOINT": "key_prefix eq 'monthlystandpoint'",
        "KTM": "key_prefix eq 'ktm'",
        "KCMA": "key_prefix eq 'kcma'",
    }

    rag_tasks = []
    active_responses = {}
    rag_prompt_tokens = 0
    rag_completion_tokens = 0

    # Launch parallel RAG queries for each active route
    for route, top_k in adjusted_top_k.items():
        filter_str = route_filters[route]
        task = run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords, filter=filter_str, top_k=top_k)
        rag_tasks.append((route, task))

    results = await asyncio.gather(*(task for _, task in rag_tasks))

    for (route, _), (response, prompt_toks, completion_toks) in zip(rag_tasks, results):
        active_responses[route] = response
        rag_prompt_tokens += prompt_toks
        rag_completion_tokens += completion_toks

    # Step 3: Run Orchestrator to combine responses
    if status:
        status["orchestrator"].markdown("🧠 Synthesizing final RAG response...")
        status["rag"].empty()

    orchestrator_sections = []

    if "MONTHLYSTANDPOINT" in active_responses:
        orchestrator_sections.append(f"""
        Information from Monthly Standpoint (monthlystandpoint) document (covering news in this month):
        {active_responses["MONTHLYSTANDPOINT"]}
        """)

    if "KTM" in active_responses:
        orchestrator_sections.append(f"""
        Information from Know the Markets (KTM) document (covering news in this quarter):
        {active_responses["KTM"]}
        """)

    if "KCMA" in active_responses:
        orchestrator_sections.append(f"""
        Information from KAsset Capital Market Assumptions (KCMA) document (published at the start of the year, covering assumptions for the whole year):
        {active_responses["KCMA"]}
        """)

    orchestrator_prompt = f"""
        Today is {today_str}.

        The information given to you are:
        {''.join(orchestrator_sections)}

        If the user's question "{user_query}" mentions specific documents, ignore what is not stated.
        Otherwise, consider all included documents, but prioritize KCMA and Monthly Standpoint over KTM in terms of correctness.
        Cross-check the facts and use those to answer the original question:
        {user_query}

        Please write your final response in a clear {language}, structured way. Make sure no important point is missed.
        """

    orchestrator_message = ChatMessageContent(role=AuthorRole.USER, content=orchestrator_prompt)
    input_tokens_orchestrator = count_tokens(orchestrator_prompt, tokenizer_4_1)

    final_response = ""
    container.markdown(final_response)  # clear container before streaming

    async for orchestration in orchestrator_agent.invoke_stream(messages=[orchestrator_message]):
        final_response += str(orchestration)
        container.markdown(final_response)
        main_thread = orchestration.thread
        has_streamed = True

    output_tokens_orchestrator = count_tokens(final_response, tokenizer_4_1)

    return (
        final_response,
        main_thread,
        input_tokens_router,
        output_tokens_router,
        rag_prompt_tokens,
        rag_completion_tokens,
        input_tokens_orchestrator,
        output_tokens_orchestrator,
        keyword_input_tokens,
        keyword_output_tokens,
        has_streamed,
    )
