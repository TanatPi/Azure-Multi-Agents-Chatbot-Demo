# 💬 Azure Semantic Kernel Chatbot

An intelligent financial assistant powered by **Azure OpenAI** and **Azure Cognitive Search** on the **Semantic Kernel** framework.  
This project tracks my work during an internship at a company in Bangkok and serves as a showcase of applied LLM agent orchestration.

> **📄 Note:** All documents used in this demo are publicly available.  
> No sensitive or proprietary information is used.
> I have permission from my direct supervisor to make this demo public.

---

## 🚀 Deployment

**Streamlit Cloud (available until late August):**  
[🔗 Live Demo](https://mm-rag-agent-demo-xil5jtaiwjk6hnbtzkkh4x.streamlit.app/)  
*(Subject to takedown if usage is high or upon request)*

**Run Locally (with your own api keys and endpoints):**
```bash
streamlit run <your-directory>/main.py
```
---

## 🧠 Functionalities
- 🤖 Uses Semantic Kernel framework with custom agent flow and orchestration (The official orchestration is said to be in early stage and is unstable).
- 📊 Support multi-modal rag (text/table/image) from unstructured data
- 📁 Cites sources with page/table/figure references and filenames
---
## User Manual

### 1. System Overview

The WIN-AI Chatbot is designed to provide accurate and insightful answers related to economic news, investment, and KAsset-specific information. It leverages OpenAI's Azure-based Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) techniques to enhance answer reliability.

#### Key Features
- Intent-based routing: The system identifies user intents such as NEWS, CALLCENTER, FUNDFACT, or BYPASS (general Q&A).
- Router Agents direct queries to the most relevant submodules.
- Keyword extraction improves document retrieval precision.
- Supports searching across text, tables, and images via Azure Cognitive Search.
- Orchestrator Agents synthesize multiple partial answers into a coherent final response.
- Streaming responses in real-time via a Streamlit web interface.

#### Architecture Overview
```text
User Query
    ↓
Main Router Agent (Detects Intent & Language)
    ↓
├── NEWS       → News Agents Flow (Router, Keyword Extractor, RAG Agents, Orchestrator)
├── CALLCENTER → Callcenter Agents Flow (Keyword Extractor, RAG Agent)
├── FUNDFACT   → Fundfact Agents Flow (Keyword Extractor, RAG Agents, Orchestrator)
└── BYPASS     → Reply Agent (General Q&A)
    ↓
Final Answer (Displayed via Streamlit)
```
---

#### 📁 Code Structure and Module Overview

This section outlines the purpose of each module and how they work together in the agent pipeline.

---

##### 2.1 `main.py`  
- Entry point for the Streamlit UI.  
- Initializes all agents through the shared Kernel instance.  
- Manages session state for conversation threads and chat history.  
- Handles user input, triggers the agent processing pipeline, and streams responses back to the UI.

##### 2.2 `main_agents_logic.py`  
- Core async orchestration logic directing requests to the appropriate sub-flows based on detected intent.  
- Supports flows for `NEWS`, `CALLCENTER`, `FUNDFACT`, and general `BYPASS` reply.  
- Tracks token usage for prompt and completion to monitor usage.

---

#### 🧩 Sub-Agent Pipelines

##### 2.3 `promptflow_logics/news_agents_logic.py`  
- Handles the `NEWS` intent flow.  
- Uses a News Router Agent to select relevant document sources (`MONTHLYSTANDPOINT`, `KCMA`, `KTM`).  
- Performs keyword extraction to refine search queries.  
- Retrieves content via a multi-modal RAG agent (text, tables) and synthesizes the result using an Orchestrator Agent.

##### 2.4 `promptflow_logics/fundfact_agents_logic.py`  
- Manages the `FUNDFACT` intent.  
- Extracts keywords and invokes:
  - A **linguistic RAG agent** (searches financial facts from text).
  - A **coder RAG agent** (interprets structured data like CSVs).
- Consolidates both answers using an Orchestrator Agent into a unified financial explanation.

##### 2.5 `promptflow_logics/callcenter_agents_logic.py`  
- Handles `CALLCENTER` intent.  
- Performs keyword extraction and uses a text-based RAG agent to search a customer support knowledge base.  
- Streams results back to the user in natural language.

---

#### 🧠 Agent Modules

##### 2.6 `agents/reply_agent.py`  
- Handles the fallback `BYPASS` intent when no specialized route is chosen.  
- Provides general Q&A responses using the chat history context.

##### 2.7 `agents/mm_rag_agent.py`  
- Provides multi-modal RAG capabilities using Azure Cognitive Search.  
- Searches across text, table, and image indices from economic PDFs.  
- Used in the NEWS pipeline.

##### 2.8 `agents/txt_rag_agent.py`  
- Creates standard text-only RAG agents for Callcenter and FundFact flows.  
- Supports custom index names for each domain.

##### 2.9 `agents/router_agent.py`  
- Creates two agents:  
  - **Main Router Agent**: Determines the top-level intent (NEWS, FUNDFACT, CALLCENTER, BYPASS).  
  - **News Router Agent**: Further routes within NEWS flow to specific sources based on document type relevance.

##### 2.10 `agents/orchestrator_agent.py`  
- Used in both NEWS and FUNDFACT flows.  
- Consolidates multiple RAG agent outputs into a single coherent response.  
- Encourages structured, non-repetitive, and multi-source-aware answers.

##### 2.11 `agents/keyword_extractor_agent.py`  
- Lightweight agent used in all flows (except BYPASS).  
- Extracts search-relevant keywords from the user query using prompt-based logic.

##### 2.12 `agents/fundfact_linguistic_rag_agent.py`  
- Specialized text RAG agent for extracting financial fact narratives.

##### 2.13 `agents/fundfact_coder_rag_agent.py`  
- Uses the **Azure Assistant Agent (API v2)** for parsing structured files (e.g., CSVs).  
- Can eventually support plotting, graph generation, or code-related outputs.

---

Feel free to expand or customize this documentation as your project evolves!

---
## 💬 Example query:
User: "How's the US stock market?"

---
#### Roadmap
##### Near Future
- Fine-tune system/user prompt
- Restructure the search index if needed
- Add a new agent flow that is capable of automatically read csv/JSON with AzureAssistant Agent API v2, along with abilities to plot graph, etc.
##### Later
- I may move away from relying completely on Azure to keep the demo accessible.
---


