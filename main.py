import os
import asyncio
import json
import requests
import streamlit as st

from openai import AsyncAzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential


# === Azure OpenAI Config ===
endpoint = st.secrets['AZURE_OPENAI_RESOURCE']
deployment = st.secrets['AZURE_OPENAI_MODEL']
subscription_key = st.secrets['AZURE_OPENAI_KEY']
api_version = "2024-12-01-preview"

client = AsyncAzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

# === Helper Functions ===
async def get_embedding(text):
    def sync_post():
        response = requests.post(
            url=embedding_endpoint,
            headers=headers,
            json={"input": text}
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']

    return await asyncio.to_thread(sync_post)

async def search_query(user_query, search_client, select, top_k=10, filter=None):
    vector = await get_embedding(user_query)
    vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=6, fields="contentVector")

    results = search_client.search(
        search_text=user_query,
        vector_queries=[vector_query],
        select=select,
        top=top_k,
        filter=filter
    )

    retrieved_info = []
    for doc in results:
        retrieved_info.append({
            "page": doc.get("page", "N/A"),
            "filename": doc.get("doc_name", "unknown.txt"),
            "content": doc.get("content", ""),
            "table": doc.get("table", "N/A"),
            "figure": doc.get("figure", "N/A")
        })

    return retrieved_info
    

# === Main Chat Function ===
async def chat_with_search_agent(user_query, system_prompt = "", filter="key_prefix eq 'monthlystandpoint'", history=None):

    if history is None:
        history = []

    context, context_table, context_image = await asyncio.gather(
        search_query(user_query, search_client_text,
                    select=['content', "page", "doc_name"], top_k=4, filter=filter),

        search_query(user_query, search_client_table,
                    select=['content', "page", "table", "doc_name"], top_k=4, filter=filter),

        search_query(user_query, search_client_image,
                    select=['content', "page", "doc_name", "figure"], top_k=4, filter=filter)
    )

    # convert to JSON strings for LLM input
    import json
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    context_table_json = json.dumps(context_table, ensure_ascii=False, indent=2)
    context_image_json = json.dumps(context_image, ensure_ascii=False, indent=2)
     # Compose messages
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"""Use the following JSON context to answer the question:


                Context text data:
                {context_json}

                Context table data:
                {context_table_json}

                Context image data:
                {context_image_json}

                Question: {user_query}
                """})

    response = await client.chat.completions.create(
        messages = messages,
        temperature=0.4,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )

    return response.choices[0].message.content

# === Main Chat Function ===
async def chat_with_ochestrator_agent(user_query, monthly_reply ="", ktm_reply="", kcma_reply="", history=None):

    messages = [{"role": "system", "content": "You are a helpful and expert financial assistant."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"""
                    You are a senior analyst. Three junior agents (Monthly Standpoint, KTM, KCMA) have each answered the question "{user_query}" based on their own context. Your job is to synthesize their answers into a single, clear, and concise explanation, in Thai, citing key points from each if relevant.

                    Answer from Monthly Standpoint:
                    {monthly_reply}

                    Answer from KTM:
                    {ktm_reply}

                    Answer from KCMA:
                    {kcma_reply}

                    Please provide a complete, well-structured response for the user based on all 3 sources. Include page/table/figure references when available.
                    """}
                )
    response = await client.chat.completions.create(
        messages= messages,
        temperature=0.4,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )

    return response.choices[0].message.content



# Run the main function
if __name__ == "__main__":
    # === Azure Search Config ===
    search_endpoint = st.secrets['COG_SEARCH_ENDPOINT']
    admin_key = st.secrets['COG_SEARCH_ADMIN_KEY']

    search_client_text = SearchClient(endpoint=search_endpoint,
                                    index_name="pdf-economic-summary",
                                    credential=AzureKeyCredential(admin_key))
    search_client_image = SearchClient(endpoint=search_endpoint,
                                    index_name="pdf-economic-summary-images",
                                    credential=AzureKeyCredential(admin_key))
    search_client_table = SearchClient(endpoint=search_endpoint,
                                    index_name="pdf-economic-summary-tables",
                                    credential=AzureKeyCredential(admin_key))

    # === Embedding Config ===
    embedding_endpoint = st.secrets['AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE']
    headers = {
        "Content-Type": "application/json",
        "Authorization": st.secrets['AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE_KEY']
    }
    
    system_prompt_RAG = """You are a helpful and friendly female assistant. You must answer questions based only on the information provided (no guessing or external knowledge) and give financial advice accordingly. 

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

    # === Streamlit UI ===
    st.set_page_config(page_title="KAsset Economic News Agent", page_icon="💬", layout="wide")
    st.title("💬 KAsset Economic News Agent")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    # Chat input box
    user_query = st.chat_input("Ask me anything about KAsset reports...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                async def full_pipeline():
                    # Prepare memory for LLM (exclude system)
                    MAX_HISTORY = 6
                    truncated_history = [
                        msg for msg in st.session_state.chat_history if msg["role"] in {"user", "assistant"}
                    ][-MAX_HISTORY:]

                    monthly_reply, ktm_reply, kcma_reply = await asyncio.gather(
                        chat_with_search_agent(user_query, system_prompt_RAG, filter="key_prefix eq 'monthlystandpoint'", history = truncated_history),
                        chat_with_search_agent(user_query, system_prompt_RAG, filter="key_prefix eq 'ktm'", history = truncated_history),
                        chat_with_search_agent(user_query, system_prompt_RAG, filter="key_prefix eq 'kcma'", history = truncated_history)
                    )
                    return await chat_with_ochestrator_agent(user_query, monthly_reply, ktm_reply, kcma_reply)
                

                # Run the orchestrator agent
                assistant_reply = asyncio.run(full_pipeline())
                st.markdown(assistant_reply)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
