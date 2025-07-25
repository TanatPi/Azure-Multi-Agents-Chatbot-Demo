import asyncio
import os
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.contents.utils.author_role import AuthorRole



import sys
# Add the parent directory (work) to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#### === Helper functions ===
# === Helper to count token ===
import tiktoken

tokenizer_4o = tiktoken.encoding_for_model("gpt-4o-mini") 
tokenizer_4_1 = tiktoken.get_encoding("o200k_base") # gpt-4.1

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

# === Load Azure credentials ===
deployment = "gpt-4o-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")
# === Helper Function ===
async def run_agent(agent, query, search_keywords = None, search_tool = None):
    if search_tool is not None:
        context_text = await search_tool.search_text_content(search_keywords, filter=None, top_k=50)

        user_prompt = f"""Use the following JSON context to answer the question:

        Context text data:
        {context_text}

        Question: {query}
        """

        input_tokens = count_tokens(user_prompt,tokenizer_4o)

        user_message = ChatMessageContent(role=AuthorRole.USER, content=user_prompt)

        response_text = ""
        async for response in agent.invoke(messages=[user_message]):
            response_text = str(response)

        output_tokens = count_tokens(response_text,tokenizer_4o)
    else:
        response_text = ""
        input_tokens = count_tokens(query, tokenizer_4o)

        async for msg in agent.invoke(query):
            # Main text response
            if hasattr(msg, "content") and msg.content:
                response_text += str(msg.content)

        output_tokens = count_tokens(response_text,tokenizer_4o)
    return {
        "text": response_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

# === Final Orchestrator ===
async def get_fundfact_agent_response(user_query: str, user_thread,keyword_extractor_agent,fundfact_linguistic_rag_agent,fundfact_linguistic_search,fundfact_coder_rag_agent, ochestrator_agent, language, status, container) -> tuple[str, str]:
    # Step 1: Run Keyword Extractor
    if status:
        status["keyword"].markdown("🔍 Extracting keywords...")

    keyword_agent_user_prompt = f"Extract keywords from this query: {user_query}"
    keyword_agent_message = ChatMessageContent(role=AuthorRole.USER, content=keyword_agent_user_prompt)

    keyword_input_tokens = count_tokens(keyword_agent_user_prompt, tokenizer_4_1)
    search_keywords = user_query

    async for response in keyword_extractor_agent.invoke(messages=[keyword_agent_message], thread=user_thread):
        search_keywords = str(response)

    keyword_output_tokens = count_tokens(search_keywords, tokenizer_4_1)


    fundfact_response_task = run_agent(fundfact_linguistic_rag_agent, user_query, search_keywords, fundfact_linguistic_search)
    csv_response_task = run_agent(fundfact_coder_rag_agent, user_query)

    if status and (status is not None):
        status["keyword"].empty()
        status["rag"].markdown("📚 Running RAG agents...")
    results = await asyncio.gather(
        fundfact_response_task,
        csv_response_task,
    )
    responses = [r["text"] for r in results]

    # === Token count for search agents
    rag_prompt_tokens = sum(r["input_tokens"] for r in results)
    rag_completion_tokens = sum(r["output_tokens"] for r in results)

    if status and (status is not None): 
        status["orchestrator"].markdown("🧠 Synthesizing final RAG response...")
        status["rag"].empty()

    orchestrator_prompt = f"""You are the final assistant. Your job is to synthesize and consolidate the following three answers into a single, coherent, complete response for the user:

        Answer from text documents:
        {responses[0]}

        Answer from spreadsheet:
        {responses[1]}

        Please write your final response in a clear in {language}, structured way. Make sure no important point is missed.
        """

    orchestrator_message = ChatMessageContent(role="user", content=orchestrator_prompt)

    # === Token count for input to orchestrator
    input_tokens_orchestrator = count_tokens(orchestrator_prompt,tokenizer_4_1)

    final_response = ""

    async for orchestration in ochestrator_agent.invoke_stream(messages=[orchestrator_message]):
        final_response += str(orchestration)
        if container is not None:
            container.markdown(final_response)
        thread = orchestration.thread
        has_streamed = True

    # === Token count for output to orchestrator
    output_tokens_orchestrator = count_tokens(final_response,tokenizer_4_1)

    return (
        final_response,
        thread,
        rag_prompt_tokens,
        rag_completion_tokens,
        input_tokens_orchestrator,
        output_tokens_orchestrator, 
        keyword_input_tokens, 
        keyword_output_tokens,
        has_streamed
    )