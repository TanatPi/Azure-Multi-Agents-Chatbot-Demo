import os
import yaml
from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings
import asyncio
# not need in real st deployment
from dotenv import load_dotenv
load_dotenv()
from agents.save_and_load_azure_assistant_agent import save_agent_id,load_agent_id


# === Load Azure credentials ===
deployment = "gpt-4o-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

def get_filepath_for_filename(filename: str) -> str:
    base_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        'data',
        'fundfact_data'
    )
    return os.path.join(base_directory, filename)

prompt_filepath = base_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'agents', 'prompts.yml')
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
system_prompt = prompts["fundfact_linguistic_rag_agent_prompt"]

async def get_fundfact_linguistic_rag_agent():
    # Step 1 : Create a client with Azure config
    client = AzureAssistantAgent.create_client(
        # service_id="chat_service",
        deployment_name=deployment,
        api_key=subscription_key,
        endpoint=endpoint,
        api_version = "2024-12-01-preview",
    )

    cached_id = load_agent_id("fundfact_linguistic_rag_agent")
    if cached_id:
        try:
            definition = await client.beta.assistants.retrieve(assistant_id=cached_id)
            return AzureAssistantAgent(client=client, definition=definition)
        except Exception as e:
            print(f"⚠️ Failed to retrieve cached assistant: {e}. Recreating...")

    base_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 
        'data',
        'fundfact_data'
    )
    # ✅ List all JSON files in current folder
    filenames = [f for f in os.listdir(base_directory) if f.endswith('.json')]
            
    # Upload the files to the client
    file_ids: list[str] = []
    for path in [get_filepath_for_filename(filename) for filename in filenames]:
        with open(path, "rb") as file:
            file = await client.files.create(file=file, purpose="assistants")
            file_ids.append(file.id)

    vector_store = await client.vector_stores.create(
        name="assistant_search",
        file_ids=file_ids,
    )

    # Get the file search tool and resources
    file_search_tools, file_search_tool_resources = AzureAssistantAgent.configure_file_search_tool(
        vector_store_ids=vector_store.id
    )

    # === Create assistant definition ===
    definition = await client.beta.assistants.create(
        model=deployment,
        instructions=system_prompt,
        name="MutualFundFactSheetAgent",
        tools=file_search_tools,
        tool_resources=file_search_tool_resources,
    )

    # === Save ID for future reuse ===
    save_agent_id("fundfact_linguistic_rag_agent", definition.id)

    # Create the agent using the client and the assistant definition
    agent = AzureAssistantAgent(
        client=client,
        definition=definition,
    )

    #thread: AssistantAgentThread | None = None

    return agent
