import os

from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings

# === Load Azure credentials ===
deployment = "gpt-4o-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === Ochestrator agent system prompt ===
system_prompt = """You are a helpful and intelligent financial assistant. Your task is to take multiple assistant-generated answers and write a single, unified, well-structured response.

Instructions:
- Include all important and unique points from the provided answers.
- Avoid duplication or contradiction.
- Present the final response clearly, with bullet points or short paragraphs.
- If answers conflict, note the discrepancy.
- Do not hallucinate or add information not provided.
- Make sure to always mention sources.

Think carefully before responding. Be concise but complete."""

async def get_ochestrator_agent() -> AzureAssistantAgent:
    # Step 1: Create a client with Azure config
    client = AzureAssistantAgent.create_client(
        deployment_name=deployment,
        api_key=subscription_key,
        endpoint=endpoint,
        api_version = "2024-07-18 (Default)",
    )

    # Step 2: Create assistant definition (only once; reused during session)
    try:
        definition = await client.beta.assistants.create(
            model=deployment,
            name="ochestrator-agent",
            instructions=system_prompt,
        )
    except Exception as e:
        print("Assistant creation failed:", e)
        raise

    # Step 3: Instantiate the agent (no plugins for now)
    agent = AzureAssistantAgent(
        client=client,
        definition=definition,
        plugins=[],  # optionally pass your SK plugins here
    )

    return agent

