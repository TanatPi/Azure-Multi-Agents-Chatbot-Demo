import os
import asyncio
import json
import requests
import yaml
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential


# === Azure OpenAI & Cognitive Search Environment Variables ===
deployment = "gpt-4o-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")
embedding_endpoint = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE")
headers = {
    "Content-Type": "application/json",
    "Authorization": os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE_KEY")
}
search_endpoint = os.environ.get("COG_SEARCH_ENDPOINT")
admin_key = os.environ.get("COG_SEARCH_ADMIN_KEY")


# === Multimodal Search Plugin ===
class SearchPlugin:
    def __init__(
        self,
        text_index_name="pdf-economic-summary",
        table_index_name="pdf-economic-summary-tables",
        image_index_name="pdf-economic-summary-images",
    ):
        self.search_client_text = SearchClient(
            endpoint=search_endpoint,
            index_name=text_index_name,
            credential=AzureKeyCredential(admin_key),
        )
        self.search_client_table = SearchClient(
            endpoint=search_endpoint,
            index_name=table_index_name,
            credential=AzureKeyCredential(admin_key),
        )
        self.search_client_image = SearchClient(
            endpoint=search_endpoint,
            index_name=image_index_name,
            credential=AzureKeyCredential(admin_key),
        )
        self.embedding_endpoint = embedding_endpoint
        self.headers = headers

    async def get_embedding(self, text: str):
        def sync_post():
            response = requests.post(
                url=self.embedding_endpoint,
                headers=self.headers,
                json={"input": text},
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

        return await asyncio.to_thread(sync_post)

    async def _search(self, query, client, select, top_k=10, filter=None):
        # Get embedding vector asynchronously
        vector = await self.get_embedding(query)

        # Create vector search query
        vector_query = VectorizedQuery(
            vector=vector, k_nearest_neighbors=top_k, fields="contentVector"
        )

        # Perform vector search with optional filtering and selection
        results = client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=select,
            top=top_k,
            filter=filter,
        )

        # Serialize search results to JSON with key fields
        return json.dumps(
            [
                {
                    "page": doc.get("page", "N/A"),
                    "filename": doc.get("doc_name", "unknown.txt"),
                    "content": doc.get("content", ""),
                    "table": doc.get("table", "N/A"),
                    "figure": doc.get("figure", "N/A"),
                }
                for doc in results
            ],
            ensure_ascii=False,
            indent=2,
        )

    @kernel_function(description="Search document text content")
    async def search_text_content(
        self,
        query: Annotated[str, "User query"],
        filter=None,
        top_k=10,
    ) -> Annotated[str, "Search results"]:
        return await self._search(
            query,
            self.search_client_text,
            select=["content", "page", "doc_name"],
            filter=filter,
            top_k=top_k,
        )

    @kernel_function(description="Search table data")
    async def search_table_content(
        self,
        query: Annotated[str, "User query"],
        filter=None,
        top_k=10,
    ) -> Annotated[str, "Search results"]:
        return await self._search(
            query,
            self.search_client_table,
            select=["content", "page", "table", "doc_name"],
            filter=filter,
            top_k=top_k,
        )

    @kernel_function(description="Search image data")
    async def search_image_content(
        self,
        query: Annotated[str, "User query"],
        filter=None,
        top_k=10,
    ) -> Annotated[str, "Search results"]:
        return await self._search(
            query,
            self.search_client_image,
            select=["content", "page", "figure", "doc_name"],
            filter=filter,
            top_k=top_k,
        )


# === Load system prompts ===
prompt_filepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "agents", "prompts.yml"
)
with open(prompt_filepath, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

system_prompt = prompts["mm_rag_agent_prompt"]


# === Create Multimodal RAG Agent ===
def get_mm_rag_agent(kernel: Kernel) -> ChatCompletionAgent:
    if "rag_agent" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                service_id="rag_agent",
                deployment_name=deployment,
                api_key=subscription_key,
                endpoint=endpoint,
            )
        )

    settings = AzureChatPromptExecutionSettings(
        service_id="rag_agent",
        temperature=0.1,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name="searchservice-mm-rag-agent",
        instructions=system_prompt,
    )


# === Create and return multimodal search plugin ===
def get_mm_search_plugin(
    text_index_name="pdf-economic-summary",
    table_index_name="pdf-economic-summary-tables",
    image_index_name="pdf-economic-summary-images",
):
    return SearchPlugin(
        text_index_name=text_index_name,
        table_index_name=table_index_name,
        image_index_name=image_index_name,
    )
