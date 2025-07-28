import os
import yaml

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

# === Azure OpenAI Configuration ===
deployment = "gpt-4.1-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === Load all prompts from YAML ===
prompt_filepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "agents",
    "prompts.yml"
)
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

# === General Router Agent Factory ===
def get_router_agent(kernel: Kernel, agent_name: str) -> ChatCompletionAgent:
    if "router_service" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                service_id="router_service",
                deployment_name=deployment,
                api_key=subscription_key,
                endpoint=endpoint,
            )
        )

    system_prompt = prompts.get(f"{agent_name}_prompt", "")
    settings = AzureChatPromptExecutionSettings(
        service_id="router_service",
        temperature=0.1,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name=agent_name,
        instructions=system_prompt,
    )
