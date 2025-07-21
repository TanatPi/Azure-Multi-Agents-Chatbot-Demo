import os

from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings

# === Load Azure credentials ===
deployment = "gpt-4.1-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === Orchestrator agent system prompt ===
system_prompt =  f"""You are the world's top investment market analyst with access to comprehensive data collected from specialized fund managers. Your task is to summarize this information clearly and concisely so that fund clients can easily understand the key insights.

The given data:

### Instructions:
1. Review the provided investment market data carefully.
2. Extract and highlight the most important trends, risks, and opportunities relevant to fund clients.
3. Use simple, clear language, avoiding technical jargon and keep things concise to ensure accessibility.
4. Provide actionable insights or recommendations where applicable.
5. Structure the summary without separating data from different managers logically, for example:
- Market Overview
- Key Trends
- Risks and Challenges
- Opportunities
- Recommendations for Fund Clients
6. Cite reference for every information possible, along with its document name.


### Guidelines:
- Focus only on the information provided; do not add external data.
- Keep the summary concise but informative.
- Ensure the tone is professional and client-friendly.

### Output Format:
Provide the summary in natural Thai language, organized with clear headings as outlined above."""

async def get_orchestrator_agent() -> AzureAssistantAgent:
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
            name="orchestrator-agent",
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

