import streamlit as st
import asyncio

from semantic_kernel.agents import AzureAssistantAgent
from azure.core.credentials import AzureKeyCredential

# === Fusion agent system prompt ===
system_prompt = """You are a helpful and intelligent financial assistant. Your task is to take multiple assistant-generated answers and write a single, unified, well-structured response.

Instructions:
- Include all important and unique points from the provided answers.
- Avoid duplication or contradiction.
- Present the final response clearly, with bullet points or short paragraphs.
- If answers conflict, note the discrepancy.
- Do not hallucinate or add information not provided.
- Make sure to always mention sources.

Think carefully before responding. Be concise but complete."""

# === Sync wrapper for async orchestration ===
async def _build_ochestrator_agent(deployment, subscription_key, endpoint):
    client = AzureAssistantAgent.create_client(
        deployment_name=deployment,
        api_key=subscription_key,
        endpoint=endpoint,
    )

    definition = await client.beta.assistants.create(
        model=deployment,
        name="ochestrator-agent",
        instructions=system_prompt,
    )

    return AzureAssistantAgent(client=client, definition=definition, plugins=[])

@st.cache_resource
def get_ochestrator_agent():
    deployment = st.secrets["AZURE_OPENAI_MODEL"]
    subscription_key = st.secrets["AZURE_OPENAI_KEY"]
    endpoint = st.secrets["AZURE_OPENAI_RESOURCE"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(_build_ochestrator_agent(deployment, subscription_key, endpoint))
