# 💬 Azure Semantic Kernel Chatbot

An intelligent financial assistant powered by Azure OpenAI and Azure Cognitive Search on the Semantic Kernel framework. This chatbot allows users to ask questions about KAsset’s economic reports and receive insights synthesized from document text, tables, and images. 

I created this during my internship at an undisclosed company, and no sensitive data is used.

👉 **Current deployment on Streamlit Cloud does not work**

However, you can try: streamlit run <your directory>/main.py to try the application locally.

---

## 🧠 Features

- 🔍 Retrieves relevant context from multiple Azure Cognitive Search indexes from multiple information types (text, tables, images)
- 🤖 Uses Azure OpenAI to generate informative, clear responses in Thai
- 📊 Supports multi-agent analysis (Monthly Standpoint, KTM, KCMA)
- 📁 Cites sources with page/table/figure references and filenames

---
## 💬 Example query:
User: "How's the US stock market?"

---
#### Roadmap
- Fine-tune system/user prompt
- Restructure the search index if needed
---

### Documents used in this demo are publicly available.
