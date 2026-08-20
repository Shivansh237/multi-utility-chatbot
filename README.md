# 🤖 AI RAG Chatbot — LangGraph + Gemini + Streamlit

A production-ready, multi-turn AI chatbot powered by **Google Gemini**, **LangGraph**, and **Streamlit** that combines Retrieval-Augmented Generation (RAG), live web search, real-time stock prices, and arithmetic — all in a single conversational interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF RAG** | Upload any PDF and ask questions grounded in its content via FAISS vector search |
| 🌐 **Web Search** | Answers real-time questions using DuckDuckGo search |
| 📈 **Stock Prices** | Fetches live stock quotes (e.g. `AAPL`, `TSLA`) via Alpha Vantage |
| 🧮 **Calculator** | Performs arithmetic through a dedicated tool node |
| 💬 **Multi-turn Memory** | Full conversation history persisted per thread in SQLite via LangGraph checkpointing |
| 🗂️ **Multi-session** | Manage multiple independent chat sessions in the sidebar |
| 🔍 **LangSmith Tracing** | Built-in observability via LangSmith for debugging and evaluation |

---

## 🏗️ Architecture

```
Streamlit UI (streamlit_rag_frontend.py)
        │
        ▼
LangGraph Agent (langraph_rag_backend.py)
        │
   ┌────┴────┐
   │ chat_node│  ← Gemini LLM with tool-calling
   └────┬────┘
        │ tools_condition
   ┌────▼────────────────────────────┐
   │           ToolNode              │
   │  ┌──────────────────────────┐   │
   │  │  rag_tool   (FAISS RAG)  │   │
   │  │  search_tool (DuckDuckGo)│   │
   │  │  get_stock_price         │   │
   │  │  calculator              │   │
   │  └──────────────────────────┘   │
   └─────────────────────────────────┘
        │
   SQLite Checkpointer (chatbot.db)
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/My_helper.git
cd My_helper
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_google_gemini_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# LangSmith (optional — for tracing)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=Chatbot
```

### 5. Run the app

```bash
streamlit run streamlit_rag_frontend.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
My_helper/
├── streamlit_rag_frontend.py  # Streamlit UI layer
├── langraph_rag_backend.py    # LangGraph agent, tools & graph definition
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .env                       # Your local secrets (git-ignored)
└── README.md
```

---

## 🔑 API Keys Required

| Service | Purpose | Get it at |
|---|---|---|
| Google Gemini | LLM + Embeddings | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| Alpha Vantage | Stock price data | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| LangSmith *(optional)* | Tracing & observability | [smith.langchain.com](https://smith.langchain.com) |

---

## 🛠️ Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Stateful agent orchestration
- **[LangChain](https://github.com/langchain-ai/langchain)** — Tool integrations & document loaders
- **[Google Gemini](https://ai.google.dev/)** — LLM (gemini-2.5-flash) + Embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search
- **[Streamlit](https://streamlit.io/)** — Interactive web UI
- **[SQLite](https://www.sqlite.org/)** — Conversation checkpointing
- **[DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)** — Real-time web search

---

## 📝 License

MIT License — feel free to use, modify, and distribute.
