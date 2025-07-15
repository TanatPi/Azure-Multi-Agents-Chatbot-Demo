# rag_agent.py

import os
import asyncio
import json
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


# === ENV ===
deployment = os.environ.get("AZURE_OPENAI_MODEL")
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")
embedding_endpoint = os.environ.get('AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE')
headers = {
    "Content-Type": "application/json",
    "Authorization": os.environ.get('AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE_KEY')
}
search_endpoint = os.environ.get('COG_SEARCH_ENDPOINT')
admin_key = os.environ.get('COG_SEARCH_ADMIN_KEY')


# === Search Plugin ===
class SearchPlugin:
    def __init__(self, text_index_name = "pdf-economic-summary", table_index_name = "pdf-economic-summary-tables", image_index_name="pdf-economic-summary-images"):
        self.search_client_text = SearchClient(
            endpoint=search_endpoint,
            index_name=text_index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.search_client_table = SearchClient(
            endpoint=search_endpoint,
            index_name=table_index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.search_client_image = SearchClient(
            endpoint=search_endpoint,
            index_name=image_index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.embedding_endpoint = embedding_endpoint
        self.headers = headers

    async def get_embedding(self, text):
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
                "page": doc.get("page", "N/A"),
                "filename": doc.get("doc_name", "unknown.txt"),
                "content": doc.get("content", ""),
                "table": doc.get("table", "N/A"),
                "figure": doc.get("figure", "N/A")
            } for doc in results
        ], ensure_ascii=False, indent=2)

    @kernel_function(description="Search document text content")
    async def search_text_content(self, query: Annotated[str, "User query"], filter=None) -> Annotated[str, "Search results"]:
        return await self._search(query, self.search_client_text, select=["content", "page", "doc_name"], filter=filter)

    @kernel_function(description="Search table data")
    async def search_table_content(self, query: Annotated[str, "User query"], filter=None) -> Annotated[str, "Search results"]:
        return await self._search(query, self.search_client_table, select=["content", "page", "table", "doc_name"], filter=filter)

    @kernel_function(description="Search image data")
    async def search_image_content(self, query: Annotated[str, "User query"], filter=None) -> Annotated[str, "Search results"]:
        return await self._search(query, self.search_client_image, select=["content", "page", "figure", "doc_name"], filter=filter)


# === System Prompt ===
system_prompt_RAG = f"""You are a helpful and friendly female assistant. You must answer questions based only on the information provided (no guessing or external knowledge) and give financial advice accordingly. 

            Instructions:
            - Use simple, clear Thai (or English) that is easy to understand for native Thai speakers.
            - If the answer has multiple supporting points, use bullet points.
            - Do not omit any number.
            - Make sure to cite which page the information is from.
            - If the info is in a table, also refer to the table number. (Table in Thai is ตาราง)
            - Make sure to include all information from various page in your answer.
            - อเมริกา, สหรัฐฯ and สหรัฐ in this context is the same country.
            - Document name can be found in 'filename' field of the given information.
            - If information from image and text coexist, includes its explanation if it is supporting the argument.

            Example:
            input 'How is Thai Economy?' 
            output
            'According to page 6 table 4 and page 11 figure 1, Thai economy is likely to go under recession due to:
            - Thailand’s Economy in Q1/2025 Grew 3.1% YoY, mainly supported by a significant surge in exports, which expanded by 13.8%. This growth was driven by many trading partners accelerating imports from Thailand ahead of the implementation of the U.S. tariff hike. However, Thailand’s economic outlook remains highly uncertain. Close attention must be paid to the outcomes of tariff negotiations with the U.S., both for Thailand and other countries in the region, in order to assess competitiveness. Additionally, domestic demand remains fragile, and tourism is beginning to show signs of slowing, posing further risks to the economy.
            - Although Q1 earnings improved compared to the previous quarter, they were flat compared to the same period last year. Sectors that outperformed expectations include ICT, agriculture and food, banking, and electronics. The market still expects listed company profits to grow 20% this year, but we anticipate downside risks to earnings ahead due to the economic slowdown.
            - In the short term, the figure 1 from page 12 supports that the SET Index faces pressure from economic uncertainties and increasing political instability. However, at current levels, downside risks to the SET Index appear limited, as it is trading at a low level comparable to during the COVID-19 period.

            We maintain our recommendation to focus portfolios on high-dividend stocks, as they offer consistent returns and long-term growth potential. This strategy is suitable in the current environment of high economic uncertainty driven by multiple factors discussed above.

            Reference: Page 6 table 4, page 11 figure 1, and  of monthly-summary.pdf.
            Do you have anything else to ask?'
            """

# === Agent & Plugin Constructor ===
def get_mm_rag_agent():
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="chat_service",
            deployment_name=deployment,
            api_key=subscription_key,
            endpoint=endpoint,
        )
    )
    settings = AzureChatPromptExecutionSettings(
        service_id="chat_service",
        temperature=0.4,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    agent = ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name="searchservivce-mm-rag-agent",
        instructions=system_prompt_RAG,
    )
    return agent

def get_search_plugin(text_index_name = "pdf-economic-summary", table_index_name = "pdf-economic-summary-tables", image_index_name="pdf-economic-summary-images"):
    return SearchPlugin(text_index_name = text_index_name, table_index_name = table_index_name, image_index_name=image_index_name)
