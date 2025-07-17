import os

from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings

# === Load Azure credentials ===
deployment = "gpt-4.1-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === Ochestrator agent system prompt ===
system_prompt = """You are a financial assistant. Combine the assistant responses into one concise, well-structured answer.

Instructions:
- Keep all key unique points.
- Avoid repeats or contradictions.
- Use bullet points or short paragraphs.
- If there's a conflict, mention it.
- Only use provided info. Do not guess.
- Cite sources if included.

Be efficient and clear."""

async def get_ochestrator_agent() -> AzureAssistantAgent:
    # Step 1: Create a client with Azure config
    client = AzureAssistantAgent.create_client(
        deployment_name=deployment,
        api_key=subscription_key,
        endpoint=endpoint,
        api_version = "2024-12-01-preview",
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

