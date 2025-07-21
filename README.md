# 💬 Azure Semantic Kernel Chatbot

An intelligent financial assistant powered by Azure OpenAI and Azure Cognitive Search on the Semantic Kernel framework. I aim to create this to track and make available my current project during my internship at a company in Bangkok.

### Documents used in this demo are publicly available.
Despite this project being a part of my internship, I want to declare that no sensitive data is used.

👉 Current deployment on Streamlit Cloud: https://mm-rag-agent-demo-xil5jtaiwjk6hnbtzkkh4x.streamlit.app/ (available until late August when my internship ends or until I decide/am requested to take it down if the usage is high)
You can also try running the application locally by using the command streamlit run <your directory>/main.py, but you have to provide the script with your API keys and endpoints.

---

## Current Features
- Answer questions you would ask a bank call center.
- Answer about economic news.

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
- I may move away from relying completely on Azure to keep the demo accessible
---


