# 🦜 LangChain Docs Assistant

An AI-powered assistant built using **LangChain, LangGraph, and Streamlit** that helps users explore documentation, support articles, and concepts related to LangChain ecosystem.

This project is an AI-powered assistant that helps users explore LangChain, LangGraph, and LangSmith documentation.
It uses LangGraph to orchestrate an intelligent agent capable of tool calling and dynamic decision-making. 
The assistant integrates documentation search, support article retrieval, and link validation tools. 
Built with Streamlit and Groq LLM, it provides a chat-based interface for fast, contextual, and reliable answers.

---

## 🚀 Features

* 💬 Chat-based interface (Streamlit)
* 🧠 Intelligent agent using LangGraph
* 🔎 Documentation search (LangChain Docs)
* 📚 Support article retrieval
* 🔗 Link validation tool
* 🛡️ Guardrails for safe and scoped responses
* ⚡ Async execution for performance
* 🔁 Tool-calling agent with fallback handling

---

## 🏗️ Architecture

```
User (Streamlit UI)
        ↓
LangGraph Agent (docs_graph.py)
        ↓
 ┌───────────────┬──────────────────┬──────────────┐
 │ Docs Tool     │ Support Tool     │ Link Checker │
 │               │                  │              │
 └───────────────┴──────────────────┴──────────────┘
        ↓
      LLM (Groq / LLaMA)
        ↓
   Final Response
```

---

## 📂 Project Structure

```
.
├── agent/
│   ├── config.py
│   ├── docs_graph.py
│   └── graph.py
│
├── tools/
│   ├── docs_tools.py
│   ├── support_tools.py
│   └── link_check_tools.py
│
├── middleware/
│   ├── guardrails_middleware.py
│   └── retry_middleware.py
│
├── prompts/
│   └── docs_agent_prompt.py
│
├── memory/
│   └── redis_memory.py
│
├── app.py
├── settings.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/langchain-docs-assistant.git
cd langchain-docs-assistant
```

---

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🧠 How It Works

### 1. User Query

User enters a question in Streamlit UI.

### 2. LangGraph Agent

* Applies guardrails
* Decides whether to call tools
* Routes execution dynamically

### 3. Tool Execution

Agent may call:

* `SearchDocsByLangChain`
* `search_support_articles`
* `get_article_content`
* `check_links`

### 4. Response Generation

* Tool results passed back to LLM
* Final response returned to UI
* Fallback used if tools fail

---

## 🧪 Testing

Use the provided test cases to validate:

* Tool execution
* Guardrails
* Fallback handling
* Error scenarios

Example:

```
What is LangChain?
How does streaming work in LangGraph?
Check if this link is valid https://docs.langchain.com
Tell me a joke
```

---

## ⚠️ Known Issues

* Tool responses may return empty if APIs are not configured
* Rate limits may occur with Groq free tier
* Prompt formatting must escape `{}` characters
* Incorrect tool binding can cause `StructuredTool` errors

---

## 🛠️ Tech Stack

* **LangChain**
* **LangGraph**
* **Groq (LLM)**
* **Streamlit**
* **Python 3.11+**

---

## 🔮 Future Improvements

* ✅ Add RAG (vector database)
* ✅ Improve tool reliability
* ✅ Add LangSmith tracing
* ✅ Streaming responses
* ✅ Better UI/UX

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a PR.

---

## 📄 License

MIT License

---

## 👤 Author

Built by **Akshay Yelne**

---

## ⭐ Support

If you like this project, please ⭐ the repo!
