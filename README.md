# 🤖 Multi Utility Chatbot — LangGraph + Gemini + Streamlit

An AI agent that chats with your PDFs, searches the web, fetches live stock prices, and does math, all in one chat window. Built with **LangGraph** for the agent loop, **Google Gemini** as the LLM, **FAISS** for PDF search, and **Streamlit** for the UI.

**🔗 Live demo:** _coming soon_ <!-- replace with your Render URL, e.g. https://multi-utility-chatbot.onrender.com -->

<!-- Add a demo GIF here once deployed:
![Demo](assets/demo.gif)
-->

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Chat with your PDF (RAG)** | Upload a PDF and ask questions about it. The text is split into chunks, embedded with Gemini, and searched with FAISS |
| 🌐 **Web search** | Answers questions about current events using DuckDuckGo |
| 📈 **Stock prices** | Live stock quotes (e.g. `AAPL`, `TSLA`) via Alpha Vantage |
| 🧮 **Safe calculator** | Evaluates full math expressions like `(3+4)*2^5` or `sqrt(144)/3` with numexpr — no `eval()`, so it can't run harmful code |
| 🤖 **Agent decides the tool** | Gemini picks which tool to use (or none) for each question, using LangGraph's tool-calling loop |
| 💬 **Chat memory** | Every conversation is saved in SQLite with LangGraph's checkpointer, so old chats survive an app restart |
| 🏷️ **Auto chat titles** | Like ChatGPT: after the first message, Gemini writes a short title for the chat (e.g. *"Tesla Stock Price Query"*). If that fails, the first message is used as the title |
| ⚡ **Streaming answers** | Answers appear word by word, with a status box showing which tool is running |
| 🔑 **Bring your own key** | Visitors can paste their own Gemini API key in the sidebar. Without one, they get a limited number of free messages |
| 🛡️ **Friendly errors** | Rate limits, invalid API keys, and scanned (text-less) PDFs show a clear message instead of crashing the app |
| 🔍 **LangSmith tracing** | Optional tracing of every agent step for debugging |

---

## 🏗️ Architecture

```
            Streamlit UI  (streamlit_rag_frontend.py)
   sidebar: chats + titles, PDF upload, own API key, free-message counter
                              │
                              ▼
           LangGraph agent  (langraph_rag_backend.py)

        START ──► chat_node ──(tool call?)──► tools ──┐
                  (Gemini)         │                  │
                     ▲             │ no               │
                     │             ▼                  │
                     │            END                 │
                     └────────────────────────────────┘

   tools:  rag_tool (FAISS)  ·  DuckDuckGo search  ·  get_stock_price  ·  calculator
                              │
                              ▼
        SQLite (chatbot.db): chat history (checkpoints) + chat titles
```

**How one question flows through the app:**

1. You type a question. Streamlit sends it to the LangGraph agent with the chat's `thread_id`.
2. `chat_node` sends the full conversation (plus a system prompt with today's date) to Gemini.
3. If Gemini wants a tool, for example the calculator, LangGraph runs it in the `tools` node and sends the result back to `chat_node`.
4. This repeats until Gemini answers in plain text. The answer streams to the UI.
5. LangGraph's checkpointer saves the conversation in SQLite after every step.

**Design decisions worth noting:**

- **`thread_id` is injected into `rag_tool` from the run config**, not passed by the LLM. The model can't pick the wrong document by mistake.
- **Today's date is in the system prompt.** Without it, the model trusted its old training data over fresh web search results, for example answering with the 2022 World Cup in 2026.
- **The calculator uses numexpr instead of `eval()`.** `eval()` runs any Python code; numexpr only understands math.

---

## 🚀 Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Shivansh237/multi-utility-chatbot.git
cd multi-utility-chatbot
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

### 4. Add your API keys

```bash
cp .env.example .env
```

Then open `.env` and fill in at least `GEMINI_API_KEY` (see [Environment variables](#️-environment-variables)).

### 5. Run the app

```bash
streamlit run streamlit_rag_frontend.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Try asking:**
- *"What is (25 * 4) + sqrt(144) / 3?"* → calculator
- *"What is the current stock price of AAPL?"* → stock tool
- *"Search the web: latest news about AI"* → web search
- Upload a PDF, then *"Summarize this document"* → RAG

---

## ☁️ Deploy on Render (free)

This repo includes a [`render.yaml`](render.yaml) file, so Render can set everything up automatically.

1. Push this project to a GitHub repository (make sure `.env` is **not** committed — it's in `.gitignore`).
2. Go to [render.com](https://render.com) → **New +** → **Blueprint** → select your repository.
3. Render reads `render.yaml` and asks for the secret values. Enter:
   - `GEMINI_API_KEY`
   - `ALPHA_VANTAGE_API_KEY`
4. Click **Apply**. The first build takes a few minutes. Then open the `.onrender.com` URL.
5. Paste that URL into the **Live demo** line at the top of this README.

`render.yaml` also sets `PUBLIC_MODE=true` (visitors only see their own chats) and `FREE_MESSAGE_LIMIT=10`.

> **Note:** On Render's free plan the app sleeps after ~15 minutes without visitors, so the first visit can take 30–60 seconds to load. The disk is also temporary, so chats and PDF indexes reset when the app restarts.

---

## ⚙️ Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Google Gemini key for the LLM and embeddings — [get one free](https://aistudio.google.com/app/apikey) |
| `ALPHA_VANTAGE_API_KEY` | For stock prices | [Get one free](https://www.alphavantage.co/support/#api-key) (about 25 requests/day) |
| `GEMINI_MODEL` | No | Gemini model name (default `gemini-3.6-flash`) |
| `FREE_MESSAGE_LIMIT` | No | Messages a visitor can send with your key before they must add their own (default `10`) |
| `PUBLIC_MODE` | No | `true` = each visitor only sees their own chats in the sidebar (default `false`) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT` | No | Turn on LangSmith tracing — [smith.langchain.com](https://smith.langchain.com) |

---

## 📁 Project structure

```
multi-utility-chatbot/
├── streamlit_rag_frontend.py  # Streamlit UI: chat, sidebar, PDF upload, own-key box
├── langraph_rag_backend.py    # LangGraph agent, tools, PDF ingestion, chat titles
├── requirements.txt           # Python dependencies
├── render.yaml                # Render deployment config
├── .env.example               # Template for your .env file
├── .env                       # Your secrets (git-ignored)
├── chatbot.db                 # SQLite database, created automatically (git-ignored)
└── README.md
```

---

## 🛠️ Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — agent loop (chat node ↔ tool node) and SQLite checkpointing
- **[LangChain](https://github.com/langchain-ai/langchain)** — tools, PDF loader, text splitter
- **[Google Gemini](https://ai.google.dev/)** — LLM (`gemini-3.6-flash`) and embeddings (`gemini-embedding-001`)
- **[FAISS](https://github.com/facebookresearch/faiss)** — vector similarity search for PDF chunks
- **[Streamlit](https://streamlit.io/)** — web UI with streaming
- **[SQLite](https://www.sqlite.org/)** — chat history and chat titles
- **[numexpr](https://github.com/pydata/numexpr)** — safe math evaluation
- **[DuckDuckGo Search](https://pypi.org/project/ddgs/)** — web search
- **[Render](https://render.com)** — hosting

---

## ⚠️ Limitations & future improvements

### Current limitations

- **PDF index is kept in memory.** Chat history is saved in SQLite and survives a restart, but the FAISS index for uploaded PDFs lives in RAM. After a restart you have to upload the PDF again.
- **One PDF per chat.** Uploading a second PDF in the same chat replaces the first one.
- **Free hosting resets data.** On Render's free plan the disk is temporary, so chat history and indexes reset after inactivity or a redeploy.
- **Free message limit is per browser session.** Refreshing the page resets the counter. It protects against accidental overuse, not against someone determined to abuse it.
- **API limits.** Gemini's free tier allows only a few requests per minute, and each question uses about 2. The free Alpha Vantage key allows about 25 requests per day.

### Planned improvements

- **Persistent vector store.** Save the FAISS index to disk (`save_local` / `load_local`) for a single user, or move to a vector database such as ChromaDB or pgvector for a multi-user deployment.
- **Managed database.** Move chat history from SQLite to Postgres so it survives redeploys.
- **Source citations.** Show the page number of each retrieved chunk in the answer (already available in the chunk metadata).
- **RAG evaluation.** Measure answer quality with RAGAS / LangSmith evaluations and track improvements.
- **API layer.** Add a FastAPI backend so other frontends can use the agent.

---

## 📝 License

MIT License — feel free to use, modify, and distribute.
