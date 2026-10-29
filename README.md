# Multi Utility Chatbot

A chatbot built with LangGraph and Google Gemini that can answer questions about a PDF you upload, search the web, look up stock prices and solve math expressions. The agent decides by itself which tool to use for each question. The UI is made with Streamlit.

**Live demo:** https://multi-utility-chatbot-3rxj.onrender.com (free instance, the first load can take up to a minute)

<!-- ![Demo](assets/demo.gif) -->

## Features

- **Chat with a PDF (RAG):** upload a PDF, it gets split into chunks, embedded with Gemini embeddings and stored in a FAISS index. Questions about the document are answered from the most relevant chunks.
- **Web search** using DuckDuckGo, for questions about recent events.
- **Stock prices** from the Alpha Vantage API (e.g. AAPL, TSLA).
- **Calculator** that evaluates whole expressions like `(3+4)*2^5` or `sqrt(144)/3`. It uses numexpr instead of `eval()`, so it can't execute arbitrary code.
- **Chat history** saved in SQLite through LangGraph's checkpointer, so old conversations are still there after restarting the app.
- **Chat titles:** after the first message Gemini generates a short title for the chat (like ChatGPT does). If that call fails, the first message is used as the title.
- **Streaming responses**, with a status box that shows which tool is being used.
- **Own API key option:** visitors of the demo can enter their own Gemini key. Without one they get a limited number of free messages.
- Error handling for rate limits, invalid keys and scanned PDFs without text, so the app shows a message instead of crashing.
- Optional tracing with LangSmith.

## How it works

```
Streamlit UI (streamlit_rag_frontend.py)
        |
        v
LangGraph agent (langraph_rag_backend.py)

   START -> chat_node -> tool call? --yes--> tools --+
              ^              |                       |
              |              no -> END               |
              +--------------------------------------+

   tools: rag_tool (FAISS), DuckDuckGo search, get_stock_price, calculator
        |
        v
SQLite (chatbot.db): chat history + chat titles
```

1. The user's message is sent to the graph together with the chat's `thread_id`.
2. `chat_node` sends the conversation and a system prompt to Gemini.
3. If Gemini asks for a tool, the `tools` node runs it and the result goes back to `chat_node`.
4. This repeats until Gemini gives a normal text answer, which is streamed to the UI.
5. The checkpointer saves the state in SQLite after every step.

Some things I ran into while building it:

- At first the LLM had to pass the `thread_id` to `rag_tool` itself, and it didn't always do it correctly. Now the tool reads it from the run config.
- Web search answers were sometimes outdated, because the model trusted its training data more than the search results. Adding today's date to the system prompt fixed it.
- The first version of the calculator only took two numbers and an operation. I switched to numexpr so it handles full expressions without the security problems of `eval()`.

## Running locally

```bash
git clone https://github.com/Shivansh237/multi-utility-chatbot.git
cd multi-utility-chatbot

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your keys (at least `GEMINI_API_KEY`). Then start the app:

```bash
streamlit run streamlit_rag_frontend.py
```

It opens at http://localhost:8501.

Some example questions:
- "What is (25 * 4) + sqrt(144) / 3?"
- "What is the current stock price of AAPL?"
- "Search the web for the latest AI news"
- upload a PDF and ask "Summarize this document"

## Deployment (Render)

The repo has a `render.yaml`, so it can be deployed as a Render Blueprint:

1. On render.com go to **New > Blueprint** and select this repository.
2. Enter `GEMINI_API_KEY` and `ALPHA_VANTAGE_API_KEY` when asked.
3. Click **Apply** and wait for the build to finish.

`render.yaml` also sets `PUBLIC_MODE=true` (each visitor only sees their own chats) and `FREE_MESSAGE_LIMIT=10`.

On the free plan the service sleeps after about 15 minutes without traffic, so the first request can take up to a minute. The disk is not persistent either, so chats are lost when the service restarts.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | yes | Gemini key for the LLM and embeddings ([get one](https://aistudio.google.com/app/apikey)) |
| `ALPHA_VANTAGE_API_KEY` | for stock prices | [get one](https://www.alphavantage.co/support/#api-key), free tier is about 25 requests/day |
| `GEMINI_MODEL` | no | model name, default `gemini-3.6-flash` |
| `FREE_MESSAGE_LIMIT` | no | free messages per visitor without their own key, default `10` |
| `PUBLIC_MODE` | no | `true` = visitors only see their own chats, default `false` |
| `LANGSMITH_*` | no | LangSmith tracing settings |

## Project structure

```
multi-utility-chatbot/
├── streamlit_rag_frontend.py   # Streamlit UI
├── langraph_rag_backend.py     # LangGraph agent, tools, PDF ingestion, chat titles
├── requirements.txt
├── render.yaml                 # Render deployment config
├── .env.example
└── README.md
```

## Tech stack

LangGraph, LangChain, Google Gemini (`gemini-3.6-flash`, `gemini-embedding-001`), FAISS, Streamlit, SQLite, numexpr, DuckDuckGo search, Alpha Vantage, Render.

## Limitations / what I want to improve

**Current limitations**

- The FAISS index is only kept in memory. Chat history survives a restart, but the PDF has to be uploaded again.
- Only one PDF per chat. A new upload replaces the old one.
- The free message counter is per browser session, so refreshing the page resets it.
- Gemini's free tier has low request limits, and one question usually takes 2 requests (one for the tool call, one for the answer).

**Next steps**

- Save the FAISS index to disk (`save_local` / `load_local`), or use a vector database like ChromaDB or pgvector when there are multiple users.
- Move chat history from SQLite to Postgres so it isn't lost on redeploys.
- Show page numbers as sources in the answers (they are already in the chunk metadata).
- Evaluate the RAG answers with RAGAS or LangSmith.
- Add a FastAPI backend so the agent can be used from other frontends.

## License

MIT
