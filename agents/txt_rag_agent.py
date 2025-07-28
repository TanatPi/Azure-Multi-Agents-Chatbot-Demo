import os
import json
import yaml
import asyncio
import requests
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential


# === Azure Environment Configuration ===
deployment = "gpt-4o-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")

embedding_endpoint = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE")
embedding_headers = {
    "Content-Type": "application/json",
    "Authorization": os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE_KEY")
}

search_endpoint = os.environ.get("COG_SEARCH_ENDPOINT")
admin_key = os.environ.get("COG_SEARCH_ADMIN_KEY")


# === Azure Cognitive Search Plugin ===
class SearchTextPlugin:
    def __init__(self, text_index_name="callcenterinfo"):
        self.search_client_text = SearchClient(
            endpoint=search_endpoint,
            index_name=text_index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.embedding_endpoint = embedding_endpoint
        self.headers = embedding_headers

    async def get_embedding(self, text: str):
        """Call Azure OpenAI embedding model to get a vector for the input text."""
        def sync_post():
            response = requests.post(
                url=self.embedding_endpoint,
                headers=self.headers,
                json={"input": text}
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        return await asyncio.to_thread(sync_post)

    async def _search(self, query, client, select, top_k=10, filter=None):
        """Perform vector-based search on the Azure Cognitive Search index."""
        vector = await self.get_embedding(query)
        vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=top_k, fields="contentVector")
        results = client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=select,
            top=top_k,
            filter=filter,
        )
        return json.dumps([
            {
                "id": doc.get("id", ""),
                "content": doc.get("content", "")
            }
            for doc in results
        ], ensure_ascii=False, indent=2)

    @kernel_function(description="Search document text content")
    async def search_text_content(
        self,
        query: Annotated[str, "User query"],
        filter=None,
        top_k=10
    ) -> Annotated[str, "Search results"]:
        """Semantic kernel function wrapper to perform content search."""
        return await self._search(query, self.search_client_text, select=["content", "id"], filter=filter, top_k=top_k)


# === Load RAG Prompt from YAML ===
prompt_filepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'agents', 'prompts.yml'
)
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)


# === Chat Agent Constructor ===
def get_txt_rag_agent(kernel: Kernel, agent_name: str) -> ChatCompletionAgent:
    # Add AzureChatCompletion service only if not already added
    if "rag_agent" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                service_id="rag_agent",
                deployment_name=deployment,
                api_key=subscription_key,
                endpoint=endpoint,
            )
        )

    # Set prompt execution settings
    settings = AzureChatPromptExecutionSettings(
        service_id="rag_agent",
        temperature=0.4,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Load agent-specific prompt
    system_prompt = prompts.get(agent_name + "_prompt", "")

    # Return RAG-enabled agent
    return ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name=agent_name,
        instructions=system_prompt,
    )


# === Search Plugin Constructor ===
def get_txt_search_plugin(text_index_name="callcenterinfo") -> SearchTextPlugin:
    return SearchTextPlugin(text_index_name=text_index_name)
