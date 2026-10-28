import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from langraph_rag_backend import (
    build_chatbot,
    chatbot,
    generate_title,
    get_chat_title,
    has_title,
    ingest_pdf,
    retrieve_all_threads,
    save_title,
    thread_document_metadata,
)

FREE_MESSAGE_LIMIT = int(os.getenv("FREE_MESSAGE_LIMIT", "10"))
PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").lower() == "true"


# ---------- helpers ----------
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def extract_text(content):
    # Gemini sometimes returns content as a list of blocks like
    # [{"type": "text", "text": "..."}] instead of a plain string
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
            elif isinstance(block, str):
                text += block
        return text
    return ""


def error_message(e):
    err = str(e)
    if "RESOURCE_EXHAUSTED" in err or "429" in err:
        return "Too many requests right now (Gemini free tier limit). Please wait a minute and try again."
    if "API_KEY_INVALID" in err or "API key not valid" in err:
        return "This Gemini API key is not valid. Please check the key in the sidebar."
    return f"Something went wrong: {err}"


# ---------- session state ----------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # on the public demo every visitor only sees the chats from their own session
    st.session_state["chat_threads"] = [] if PUBLIC_MODE else retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "free_messages_used" not in st.session_state:
    st.session_state["free_messages_used"] = 0

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ---------- sidebar ----------
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Current chat:** {get_chat_title(thread_key)}")

if st.sidebar.button("+ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

with st.sidebar.expander("Use your own Gemini key"):
    user_api_key = st.text_input(
        "Gemini API key (optional)",
        type="password",
        help="Get a free key at aistudio.google.com. It is only used for this session and never saved.",
    ).strip()

if user_api_key:
    st.sidebar.caption("Using your own key, no message limit.")
else:
    left = max(FREE_MESSAGE_LIMIT - st.session_state["free_messages_used"], 0)
    st.sidebar.caption(f"Free messages left: {left} / {FREE_MESSAGE_LIMIT}")

active_chatbot = build_chatbot(user_api_key) if user_api_key else chatbot

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

# key per thread, otherwise the old file stays in the uploader after "New Chat"
# and gets indexed again into the new chat
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", type=["pdf"], key=f"uploader-{thread_key}"
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                    api_key=user_api_key or None,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label="PDF indexed", state="complete", expanded=False)
            except Exception as e:
                status_box.update(label=f"Could not index PDF: {e}", state="error", expanded=True)

st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(
            get_chat_title(thread_id),
            key=f"side-thread-{thread_id}",
            use_container_width=True,
            type="primary" if str(thread_id) == thread_key else "secondary",
        ):
            selected_thread = thread_id

# ---------- main chat ----------
st.title("Multi Utility Chatbot")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input and not user_api_key and st.session_state["free_messages_used"] >= FREE_MESSAGE_LIMIT:
    st.warning(
        f"You have used all {FREE_MESSAGE_LIMIT} free messages. "
        "Add your own Gemini key in the sidebar to keep chatting."
    )
    user_input = None

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        run = {"status_box": None, "failed": False}

        def stream_answer():
            try:
                for message_chunk, _ in active_chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if run["status_box"] is None:
                            run["status_box"] = st.status(f"Using `{tool_name}` ...", expanded=True)
                        else:
                            run["status_box"].update(
                                label=f"Using `{tool_name}` ...", state="running", expanded=True
                            )

                    # stream_mode="messages" gives AIMessageChunk objects, not AIMessage
                    if isinstance(message_chunk, AIMessageChunk):
                        text = extract_text(message_chunk.content)
                        if text:
                            yield text
            except Exception as e:
                run["failed"] = True
                yield error_message(e)

        ai_message = st.write_stream(stream_answer())

        if run["status_box"] is not None:
            run["status_box"].update(label="Tool finished", state="complete", expanded=False)

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    # failed requests don't count towards the free limit
    if not user_api_key and not run["failed"]:
        st.session_state["free_messages_used"] += 1

    # first answer in a new chat -> generate a title, rerun so the sidebar shows it
    if len(st.session_state["message_history"]) == 2 and not has_title(thread_key):
        save_title(thread_key, generate_title(user_input, api_key=user_api_key or None))
        st.rerun()

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread

    temp_messages = []
    for msg in load_conversation(selected_thread):
        # skip tool results and AI messages that only contain a tool call,
        # they would show up as raw JSON or empty bubbles
        if not isinstance(msg, (HumanMessage, AIMessage)):
            continue
        text = extract_text(msg.content)
        if not text.strip():
            continue
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": text})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
