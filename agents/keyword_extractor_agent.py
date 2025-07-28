import os
import yaml

from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

# === Azure OpenAI Configuration ===
deployment = "gpt-4.1-nano"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === Load Prompt from YAML ===
prompt_filepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'agents', 'prompts.yml'
)
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
system_prompt = prompts["keyword_extractor_agent_prompt"]


# === Keyword Extractor Agent Constructor ===
def get_keyword_extractor_agent(kernel: Kernel) -> ChatCompletionAgent:
    # Add the Azure chat service to the kernel (if not already present)
    if "keyword_chat_service" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                service_id="keyword_chat_service",
                deployment_name=deployment,
                api_key=subscription_key,
                endpoint=endpoint,
            )
        )

    # Chat execution settings for low-temperature, deterministic response
    settings = AzureChatPromptExecutionSettings(
        service_id="keyword_chat_service",
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Create the keyword extraction agent
    agent = ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name="keyword-extractor-agent",
        instructions=system_prompt,
    )

    return agent
