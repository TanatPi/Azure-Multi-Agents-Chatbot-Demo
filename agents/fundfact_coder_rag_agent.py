import os
import yaml
import asyncio
from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings
from agents.save_and_load_azure_assistant_agent import save_agent_id,load_agent_id,save_file_id,load_file_id
# not need in real st deployment
from dotenv import load_dotenv
load_dotenv()

# === Load Azure credentials ===
deployment = "gpt-4.1-mini"
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

async def get_fundfact_coder_rag_agent(force_prompt_update=False, force_file_update=False, prompt_overide = None):
    client = AzureAssistantAgent.create_client(
        deployment_name=deployment,
        api_key=subscription_key,
        endpoint=endpoint,
        api_version="2024-12-01-preview",
    )

    assistant_name = "fundfact_coder_rag_agent"
    cached_id = load_agent_id(assistant_name)
    definition = None
    tools = None
    tool_resources = None

    # === File upload (always done if file update is forced)
    file_ids = []
    base_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data', 'fundfact_data'
    )
    filenames = [f for f in os.listdir(base_directory) if f.endswith('.csv')]

    for filename in filenames:
        path = get_filepath_for_filename(filename)
        if not force_file_update:
            cached_file_id = load_file_id(filename)
            if cached_file_id:
                file_ids.append(cached_file_id)
                continue
        with open(path, "rb") as file:
            uploaded = await client.files.create(file=file, purpose="assistants")
            file_ids.append(uploaded.id)
            save_file_id(filename, uploaded.id)

    if file_ids:
        tools, tool_resources = AzureAssistantAgent.configure_code_interpreter_tool(file_ids=file_ids)

    # === Retrieve or create assistant
    if cached_id:
        try:
            definition = await client.beta.assistants.retrieve(assistant_id=cached_id)

            # --- Conditional updates ---
            update_payload = {}
            if force_prompt_update:
                if prompt_overide is None:
                    update_payload["instructions"] = system_prompt
                else:
                    update_payload["instructions"] = prompt_overide
            if force_file_update and tools and tool_resources:
                update_payload["tools"] = tools
                update_payload["tool_resources"] = tool_resources

            if update_payload:
                definition = await client.beta.assistants.update(
                    assistant_id=cached_id, **update_payload
                )
                print(f"🔁 Assistant {assistant_name} updated.")

            return AzureAssistantAgent(client=client, definition=definition)

        except Exception as e:
            print(f"⚠️ Failed to retrieve/update assistant: {e}, recreating...")

    # === Full recreation fallback
    definition = await client.beta.assistants.create(
        model=deployment,
        name="FundFactCSVAgent",
        instructions=system_prompt,
        tools=tools,
        tool_resources=tool_resources,
    )
    save_agent_id(assistant_name, definition.id)

    return AzureAssistantAgent(client=client, definition=definition)