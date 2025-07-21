import os
import yaml
from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings

# === Load Azure credentials ===
deployment = "gpt-4.1-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === orchestrator agent system prompt ===
# === Load system prompt from YAML ===
prompt_filepath = base_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'agents', 'prompts.yml')
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
system_prompt = prompts["orchestrator_prompt"]

async def get_orchestrator_agent() -> AzureAssistantAgent:
    # Step 1: Create a client with Azure config
    client = AzureAssistantAgent.create_client(
        deployment_name=deployment,
        api_key=subscription_key,
        endpoint=endpoint,
        api_version = "2024-12-01-preview",
    )

    # Step 2: Create assistant definition (only once; reused during session)
    definition = await client.beta.assistants.create(
        model=deployment,
        name="orchestrator-agent",
        instructions=system_prompt,
    )


    # Step 3: Instantiate the agent (no plugins for now)
    agent = AzureAssistantAgent(
        client=client,
        definition=definition,
        plugins=[],  # optionally pass your SK plugins here
    )

    return agent

