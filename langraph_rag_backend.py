from __future__ import annotations
import asyncio
import math
import os
import sqlite3
import tempfile
from datetime import date
from typing import Annotated, Any, Dict, Optional, TypedDict
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import numexpr
import requests

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


def make_llm(api_key: Optional[str] = None):
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key or DEFAULT_API_KEY,
        thinking_level="minimal",
    )


def make_embeddings(api_key: Optional[str] = None):
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", google_api_key=api_key or DEFAULT_API_KEY
    )


# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No text found in this PDF (it may be a scanned image).")

        vector_store = FAISS.from_documents(chunks, make_embeddings(api_key))
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(expression: str) -> dict:
    """
    Evaluate a math expression, e.g. "(3 + 4) * 2**5" or "sqrt(16) + 10 % 3".
    Supports + - * / ** %, brackets, and functions like sqrt, sin, cos, log.
    """
    # numexpr instead of eval() so the LLM can't make us run arbitrary code
    try:
        result = numexpr.evaluate(expression.replace("^", "**")).item()
    except Exception as e:
        return {"error": f"Invalid expression '{expression}': {e}"}

    if not math.isfinite(result):  # e.g. 1.0/0 gives inf instead of raising
        return {"error": "Result is undefined (for example, division by zero)"}

    return {"expression": expression, "result": result}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage. Requires ALPHA_VANTAGE_API_KEY in environment.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is not set"}
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol.strip().upper(), "apikey": api_key},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        return {"error": f"Stock API request failed: {e}"}
    # when rate limited Alpha Vantage still returns 200, with a "Note"/"Information" field
    if "Global Quote" not in data or not data["Global Quote"]:
        return {"error": data.get("Note") or data.get("Information") or f"No quote found for '{symbol}'"}
    return data


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the PDF uploaded in this chat.
    """
    # thread_id comes from the run config, the LLM doesn't need to pass it
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Checkpointer
# -------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Nodes + Graph
# -------------------
def build_chatbot(api_key: Optional[str] = None):
    """Compile the graph with a given Gemini key. All graphs share one checkpointer."""
    llm_with_tools = make_llm(api_key).bind_tools(tools)

    def chat_node(state: ChatState, config: RunnableConfig):
        """LLM node that may answer or request a tool call."""
        system_message = SystemMessage(
            content=(
                # without the date the model trusts its training data over search results
                f"Today's date is {date.today():%d %B %Y}. "
                "You are a helpful assistant. For questions about the uploaded PDF, call "
                "the `rag_tool`. You can also use the web search, stock price, and "
                "calculator tools when helpful. If no document is available, ask the user "
                "to upload a PDF."
            )
        )

        messages = [system_message, *state["messages"]]
        response = llm_with_tools.invoke(messages, config=config)
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpointer)


chatbot = build_chatbot()

# -------------------
# 7. Helpers
# -------------------
def retrieve_all_threads():
    # checkpoints come newest first; dict keeps that order while removing duplicates
    all_threads = {}
    for checkpoint in checkpointer.list(None):
        all_threads[checkpoint.config["configurable"]["thread_id"]] = True
    return list(all_threads)[::-1]  # oldest first, the frontend reverses it


# -------------------
# 8. Chat titles
# -------------------
conn.execute("CREATE TABLE IF NOT EXISTS chat_titles (thread_id TEXT PRIMARY KEY, title TEXT)")
conn.commit()


def save_title(thread_id: str, title: str):
    conn.execute(
        "INSERT OR REPLACE INTO chat_titles (thread_id, title) VALUES (?, ?)",
        (str(thread_id), title),
    )
    conn.commit()


def has_title(thread_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM chat_titles WHERE thread_id = ?", (str(thread_id),)
    ).fetchone()
    return row is not None


def short_text(text: str, max_len: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def generate_title(first_message: str, api_key: Optional[str] = None) -> str:
    prompt = (
        "Generate a concise 3 to 5 word title for a chat that starts with this message:\n"
        f'"{first_message}"\n'
        "Return ONLY the title, no quotes or punctuation."
    )
    try:
        title = make_llm(api_key).invoke(prompt).text.strip().strip('"').strip()
        if title:
            return short_text(title)
    except Exception:
        pass  # rate limit etc., fall back to the message itself
    return short_text(first_message)


def get_chat_title(thread_id: str) -> str:
    row = conn.execute(
        "SELECT title FROM chat_titles WHERE thread_id = ?", (str(thread_id),)
    ).fetchone()
    if row:
        return row[0]

    # chats created before titles were added
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    for msg in state.values.get("messages", []):
        if msg.type == "human":
            return short_text(msg.text)
    return "New chat"


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
