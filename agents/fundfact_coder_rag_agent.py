import os
import yaml
import asyncio
from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings
from agents.save_and_load_azure_assistant_agent import save_agent_id,load_agent_id
# not need in real st deployment
from dotenv import load_dotenv
load_dotenv()

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
system_prompt = prompts["fundfact_coder_rag_agent_prompt"]

async def get_fundfact_coder_rag_agent():
    # Create Azure client
    client = AzureAssistantAgent.create_client(
        deployment_name=deployment,
        api_key=subscription_key,
        endpoint=endpoint,
        api_version="2024-12-01-preview",
    )

    # === Check for cached assistant ID
    cached_id = load_agent_id("fundfact_coder_rag_agent")
    if cached_id:
        try:
            definition = await client.beta.assistants.retrieve(assistant_id=cached_id)
            return AzureAssistantAgent(client=client, definition=definition)
        except Exception as e:
            print(f"⚠️ Failed to retrieve cached coder agent: {e}. Recreating...")

    # === Upload CSV files
    base_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        'data',
        'fundfact_data'
    )
    filenames = [f for f in os.listdir(base_directory) if f.endswith('.csv')]

    file_ids = []
    for filename in filenames:
        path = get_filepath_for_filename(filename)
        with open(path, "rb") as file:
            uploaded = await client.files.create(file=file, purpose="assistants")
            file_ids.append(uploaded.id)

    # === Create tools: Code interpreter + uploaded files
    code_interpreter_tools, code_interpreter_tool_resources = AzureAssistantAgent.configure_code_interpreter_tool(
        file_ids=file_ids
    )

    # === Create assistant definition
    definition = await client.beta.assistants.create(
        model=deployment,
        name="FundFactCSVAgent",
        instructions=system_prompt,
        tools=code_interpreter_tools,
        tool_resources=code_interpreter_tool_resources,
    )

    # === Save ID for future reuse
    save_agent_id("fundfact_coder_rag_agent", definition.id)

    # === Return full assistant agent
    return AzureAssistantAgent(client=client, definition=definition)