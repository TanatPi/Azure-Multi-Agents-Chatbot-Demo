import asyncio
import os
import sys
from pathlib import Path
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

# Add the parent directory (work) to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tiktoken

# === Tokenizers ===
tokenizer_4o = tiktoken.encoding_for_model("gpt-4o-mini")
tokenizer_4_1 = tiktoken.get_encoding("o200k_base")  # gpt-4.1

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in the given text using specified tokenizer."""
    return len(tokenizer.encode(text))

# === Azure OpenAI environment variables ===
deployment = "gpt-4o-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === File ID Storage for Assistant Context Awareness ===
FILE_ID_PATH = Path.cwd() / "agents" / "azure_assistant_file_ids.json"

def get_uploaded_file_summary() -> str:
    """
    Reads uploaded file IDs and returns a short summary in Markdown format.
    This is helpful for the assistant when it references external files.
    """
    if not FILE_ID_PATH.exists():
        return ""
    with open(FILE_ID_PATH, "r", encoding="utf-8") as f:
        file_ids = json.load(f)
    if not file_ids:
        return ""
    lines = ["You may refer to the following uploaded files:"]
    for filename, fid in file_ids.items():
        lines.append(f"- `{filename}` (file ID: `{fid}`)")
    return "\n".join(lines)

# === Helper function to run agent with optional search context ===
async def run_agent(agent, query, search_keywords=None, search_tool=None):
    if search_tool is not None:
        # Search for context data first
        context_text = await search_tool.search_text_content(search_keywords, filter=None, top_k=50)

        user_prompt = f"""Use the following JSON context to answer the question:

        Context text data:
        {context_text}

        Question: {query}
        """

        input_tokens = count_tokens(user_prompt, tokenizer_4o)
        user_message = ChatMessageContent(role=AuthorRole.USER, content=user_prompt)

        response_text = ""
        async for response in agent.invoke(messages=[user_message]):
            response_text = str(response)

        output_tokens = count_tokens(response_text, tokenizer_4o)

    else:
        # If no search is used, pass query + file reference directly
        file_id_summary = get_uploaded_file_summary()
        user_input = f"""Answer the question: {query} given the dictionary of filename : file_id stored with you are {file_id_summary}"""
        
        response_text = ""
        input_tokens = count_tokens(user_input, tokenizer_4o)
        async for msg in agent.invoke(user_input):
            if hasattr(msg, "content") and msg.content:
                response_text += str(msg.content)

        output_tokens = count_tokens(response_text, tokenizer_4o)

    return {
        "text": response_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# === Main orchestrator flow for fund fact queries ===
async def get_fundfact_agent_response(
    user_query: str,
    user_thread,
    keyword_extractor_agent,
    fundfact_linguistic_rag_agent,
    fundfact_linguistic_search,
    fundfact_coder_rag_agent,
    orchestrator_agent,
    language,
    status,
    container
) -> tuple[str, str]:
    # Step 1: Extract keywords
    if status:
        status["keyword"].markdown("🔍 Extracting keywords...")

    keyword_agent_user_prompt = f"Extract keywords from this query: {user_query}"
    keyword_agent_message = ChatMessageContent(role=AuthorRole.USER, content=keyword_agent_user_prompt)

    keyword_input_tokens = count_tokens(keyword_agent_user_prompt, tokenizer_4_1)
    search_keywords = user_query

    async for response in keyword_extractor_agent.invoke(messages=[keyword_agent_message], thread=user_thread):
        search_keywords = str(response)

    keyword_output_tokens = count_tokens(search_keywords, tokenizer_4_1)

    # Step 2: Run two RAG agents in parallel:
    # - linguistic rag agent with search context
    # - coder rag agent without search context
    fundfact_response_task = run_agent(fundfact_linguistic_rag_agent, user_query, search_keywords, fundfact_linguistic_search)
    csv_response_task = run_agent(fundfact_coder_rag_agent, user_query)

    if status:
        status["keyword"].empty()
        status["rag"].markdown("📚 Running RAG agents...")

    results = await asyncio.gather(fundfact_response_task, csv_response_task)
    responses = [r["text"] for r in results]

    # Token counts for all RAG agents
    rag_prompt_tokens = sum(r["input_tokens"] for r in results)
    rag_completion_tokens = sum(r["output_tokens"] for r in results)

    if status:
        status["orchestrator"].markdown("🧠 Synthesizing final RAG response...")
        status["rag"].empty()

    # Step 3: Orchestrator prompt to combine responses
    orchestrator_prompt = f"""You are the final assistant. Your job is to synthesize and consolidate the following three answers into a single, coherent, complete response for the user:

        Answer from text documents:
        {responses[0]}

        Answer from spreadsheet:
        {responses[1]}

        Use the answer from the spreadsheet as the **primary source** of truth, especially when the question asks about which fund invests in a specific stock, country, commodity, or sector.
        Please write your final response in a clear in {language}, structured way. Make sure no important point is missed.
        """

    orchestrator_message = ChatMessageContent(role=AuthorRole.USER, content=orchestrator_prompt)

    # Token count for orchestrator input
    input_tokens_orchestrator = count_tokens(orchestrator_prompt, tokenizer_4_1)

    final_response = ""
    thread = None
    has_streamed = False

    async for orchestration in orchestrator_agent.invoke_stream(messages=[orchestrator_message]):
        final_response += str(orchestration)
        if container is not None:
            container.markdown(final_response)
        thread = orchestration.thread
        has_streamed = True

    # Token count for orchestrator output
    output_tokens_orchestrator = count_tokens(final_response, tokenizer_4_1)

    return (
        final_response,
        thread,
        rag_prompt_tokens,
        rag_completion_tokens,
        input_tokens_orchestrator,
        output_tokens_orchestrator,
        keyword_input_tokens,
        keyword_output_tokens,
        has_streamed,
    )
