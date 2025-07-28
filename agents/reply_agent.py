import os
import yaml
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel import Kernel

# === Azure OpenAI Deployment and Credentials ===
deployment = "gpt-4.1-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

# === Load System Prompt from YAML ===
prompt_filepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'agents',
    'prompts.yml'
)
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

system_prompt = prompts.get("reply_agent_prompt", "")


# === Create and Return the Reply Agent ===
def get_reply_agent(kernel: Kernel) -> ChatCompletionAgent:
    # Add AzureChatCompletion service to kernel if not already present
    if "reply_agent" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                service_id="reply_agent",
                deployment_name=deployment,
                api_key=subscription_key,
                endpoint=endpoint,
            )
        )

    # Define prompt execution settings for this agent
    settings = AzureChatPromptExecutionSettings(
        service_id="reply_agent",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Instantiate ChatCompletionAgent with the loaded system prompt and settings
    agent = ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name="general_reply",
        instructions=system_prompt,
    )
    return agent
