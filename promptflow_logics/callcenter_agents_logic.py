import asyncio
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

# === Tokenizer setup ===
import tiktoken

tokenizer_4o = tiktoken.encoding_for_model("gpt-4o-mini")
tokenizer_4_1 = tiktoken.get_encoding("o200k_base")  # gpt-4.1

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using given tokenizer."""
    return len(tokenizer.encode(text))

# === Main call center agent flow ===
async def get_callcenter_agent_response(
    user_query: str,
    user_thread,
    txt_rag_agent,
    keyword_extractor_agent,
    txt_search,
    language,
    status,
    container
):
    # Step 1: Extract keywords from user query
    if status:
        status["keyword"].markdown("🔍 Extracting keywords...")

    keyword_prompt = f"Extract keywords from this query: {user_query}"
    keyword_message = ChatMessageContent(role=AuthorRole.USER, content=keyword_prompt)

    keyword_input_tokens = count_tokens(keyword_prompt, tokenizer_4_1)
    search_keywords = user_query

    async for response in keyword_extractor_agent.invoke(messages=[keyword_message], thread=user_thread):
        search_keywords = str(response)

    keyword_output_tokens = count_tokens(search_keywords, tokenizer_4_1)

    # Step 2: Use keywords to retrieve context and query the text RAG agent
    if status:
        status["rag"].markdown("📚 Running RAG agents...")
        status["keyword"].empty()

    context_text = await txt_search.search_text_content(search_keywords, filter=None, top_k=10)

    user_prompt = f"""Use the following JSON context to answer the question in {language}:

        Context text data:
        {context_text}

        Question: {user_query}
        """

    rag_prompt_tokens = count_tokens(user_prompt, tokenizer_4o)
    user_message = ChatMessageContent(role=AuthorRole.USER, content=user_prompt)

    response_text = ""
    main_thread = None
    has_streamed = False

    async for response in txt_rag_agent.invoke_stream(messages=[user_message]):
        response_text += str(response)
        container.markdown(response_text)
        main_thread = response.thread
        has_streamed = True

    rag_completion_tokens = count_tokens(response_text, tokenizer_4o)

    # Return all relevant information for tracking usage and threading
    return (
        response_text,
        main_thread,
        rag_prompt_tokens,
        rag_completion_tokens,
        keyword_input_tokens,
        keyword_output_tokens,
        has_streamed,
    )
