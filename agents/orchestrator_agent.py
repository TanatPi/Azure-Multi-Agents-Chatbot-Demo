import os
import yaml
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel import Kernel

# === Load Azure credentials ===
deployment = "gpt-4.1-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === Load system prompt from YAML ===
prompt_filepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'agents', 'prompts.yml'
)
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
system_prompt = prompts["orchestrator_prompt"]

# === Create orchestrator agent ===
def get_orchestrator_agent(kernel: Kernel) -> ChatCompletionAgent:
    # Add AzureChatCompletion service only if not added yet
    if "gpt-4.1-mini" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                    service_id="gpt-4.1-mini",
                    deployment_name=deployment,
                    api_key=subscription_key,
                    endpoint=endpoint,
            )
        )

    settings = AzureChatPromptExecutionSettings(
        service_id="gpt-4.1-mini",
        temperature=0.4,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    agent = ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name="orchestrator",
        instructions=system_prompt,
    )
    return agent
