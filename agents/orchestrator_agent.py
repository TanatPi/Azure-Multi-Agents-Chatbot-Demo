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


# === Create orchestrator agent ===
def get_orchestrator_agent(kernel: Kernel, agent_name: str) -> ChatCompletionAgent:
    # Add AzureChatCompletion service only if not added yet
    if "orchestrator" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                    service_id="orchestrator",
                    deployment_name=deployment,
                    api_key=subscription_key,
                    endpoint=endpoint,
            )
        )

    settings = AzureChatPromptExecutionSettings(
        service_id="orchestrator",
        temperature=0.3,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    system_prompt = prompts.get(agent_name + "_prompt", "")
    agent = ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name=agent_name,
        instructions=system_prompt,
    )
    return agent
