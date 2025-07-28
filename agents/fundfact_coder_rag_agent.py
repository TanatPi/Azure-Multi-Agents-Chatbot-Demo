import os
import yaml
import asyncio

from semantic_kernel.agents import AzureAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings

from agents.save_and_load_azure_assistant_agent import (
    save_agent_id, load_agent_id,
    save_file_id, load_file_id
)

# Not needed in real deployment, but helpful for local dev/testing
from dotenv import load_dotenv
load_dotenv()

# === Azure OpenAI configuration ===
deployment = "gpt-4.1-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === Load assistant prompt ===
prompt_filepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'agents', 'prompts.yml'
)
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
system_prompt = prompts["fundfact_coder_rag_agent_prompt"]


# === Utility to get full file path for a data file ===
def get_filepath_for_filename(filename: str) -> str:
    base_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        'data', 'fundfact_data'
    )
    return os.path.join(base_directory, filename)


# === Main function to get the CSV agent (creates or updates if needed) ===
async def get_fundfact_coder_rag_agent(
    force_prompt_update=False,
    force_file_update=False,
    prompt_overide=None
):
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

    # === Upload CSV files to assistant (if needed) ===
    file_ids = []
    base_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        'data', 'fundfact_data'
    )
    filenames = [f for f in os.listdir(base_directory) if f.endswith('.csv')]

    for filename in filenames:
        path = get_filepath_for_filename(filename)

        # Reuse uploaded file ID unless forced to update
        if not force_file_update:
            cached_file_id = load_file_id(filename)
            if cached_file_id:
                file_ids.append(cached_file_id)
                continue

        with open(path, "rb") as file:
            uploaded = await client.files.create(file=file, purpose="assistants")
            file_ids.append(uploaded.id)
            save_file_id(filename, uploaded.id)

    # Attach files as tool resources if available
    if file_ids:
        tools, tool_resources = AzureAssistantAgent.configure_code_interpreter_tool(file_ids=file_ids)

    # === Try to retrieve and update existing assistant if available ===
    if cached_id:
        try:
            definition = await client.beta.assistants.retrieve(assistant_id=cached_id)

            update_payload = {}

            # Update prompt if forced or overridden
            if force_prompt_update:
                update_payload["instructions"] = prompt_overide if prompt_overide else system_prompt

            # Update tool resources if files are re-uploaded
            if force_file_update and tools and tool_resources:
                update_payload["tools"] = tools
                update_payload["tool_resources"] = tool_resources

            # Apply update if needed
            if update_payload:
                definition = await client.beta.assistants.update(
                    assistant_id=cached_id, **update_payload
                )
                print(f"🔁 Assistant {assistant_name} updated.")

            return AzureAssistantAgent(client=client, definition=definition)

        except Exception as e:
            print(f"⚠️ Failed to retrieve/update assistant: {e}, recreating...")

    # === Create a new assistant if retrieval failed ===
    definition = await client.beta.assistants.create(
        model=deployment,
        name="FundFactCSVAgent",
        instructions=system_prompt,
        tools=tools,
        tool_resources=tool_resources,
    )
    save_agent_id(assistant_name, definition.id)

    return AzureAssistantAgent(client=client, definition=definition)
