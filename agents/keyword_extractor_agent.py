# rag_agent.py
import os
import asyncio
import yaml

from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

# === ENV ===
deployment = "gpt-4.1-nano"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")


# === System Prompt ===
# === Load system prompt from YAML ===
prompt_filepath = base_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'agents', 'prompts.yml')
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
system_prompt = prompts["keyword_extractor_agent_prompt"]

# === Agent & Plugin Constructor ===
def get_keyword_extractor_agent():
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="keyword_chat_service",
            deployment_name="gpt-4.1-nano",
            api_key=subscription_key,
            endpoint=endpoint,
        )
    )
    settings = AzureChatPromptExecutionSettings(
        service_id="keyword_chat_service",
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    agent = ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name="keyword-extractor-agent",
        instructions=system_prompt,
    )
    return agent
