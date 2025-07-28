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
User Query
↓
Main Router Agent (Detects Intent & Language)
↓
├── NEWS → News Agents Flow (Router, Keyword Extractor, RAG Agents, Orchestrator)
├── CALLCENTER → Callcenter Agents Flow (Keyword Extractor, RAG Agent)
├── FUNDFACT → Fundfact Agents Flow (Keyword Extractor, RAG Agents, Orchestrator)
└── BYPASS → Reply Agent (General Q&A)
↓
Final Answer (Displayed via Streamlit)

### 2. Code Module Documentation

#### 2.1 `main.py`  
- Entry point for the Streamlit UI.  
- Initializes all agents through the shared Kernel instance.  
- Manages session state for conversation threads and chat history.  
- Handles user input, triggers the agent processing pipeline, and streams responses back to the UI.

#### 2.2 `main_agents_logic.py`  
- Core async orchestration logic directing requests to the appropriate sub-flows based on detected intent.  
- Supports flows for NEWS, CALLCENTER, FUNDFACT, and general BYPASS reply.  
- Tracks token usage for prompt and completion to monitor usage.

#### 2.3 `news_agents_logic.py`  
- Handles the NEWS intent flow.  
- Uses a News Router Agent to select relevant document sources.  
- Performs keyword extraction to refine search queries.  
- Calls multiple RAG agents to retrieve and synthesize information, finalized by the Orchestrator Agent.

#### 2.4 `fundfact_agents_logic.py`  
- Manages the FUNDFACT intent.  
- Employs keyword extraction, linguistic RAG agent, coder RAG agent, and an orchestrator for fund-specific questions.  
- Consolidates answers from both text documents and spreadsheets into a clear, comprehensive response.

#### 2.5 `callcenter_agents_logic.py`  
- Supports CALLCENTER intent queries.  
- Extracts keywords and uses a Text RAG agent to search a customer support knowledge base and generate answers.

#### 2.6 `reply_agent.py`  
- Provides fallback/general Q&A support (BYPASS intent).  
- Uses an LLM to answer based on chat history context.

#### 2.7 `mm_rag_agent.py`  
- Handles document retrieval and summarization from economic documents using Azure Cognitive Search.  
- Supports text, table, and image indices.  
- Uses LLM for generating answers based on retrieved data.

#### 2.8 `router_agent.py`  
- Implements main and sub-router agents to detect intent and route queries accordingly.  
- Uses prompt-based logic for routing decisions.

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


