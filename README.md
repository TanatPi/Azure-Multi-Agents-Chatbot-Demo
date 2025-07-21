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

## Current Features
- Answer questions you would ask a bank call center.
- Answer about economic and investment news.

## 🧠 Functionalities
- 🤖 Uses Semantic Kernel framework with custom agent flow and orchestration (The official orchestration is said to be in early stage and is unstable).
- 📊 Support multi-modal rag (text/table/image) from unstructured data
- 📁 Cites sources with page/table/figure references and filenames

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


