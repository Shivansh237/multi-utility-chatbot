import os
import uuid 
 
import streamlit as st 

# ================= CHANGE 1 =================
# Added AIMessageChunk because chatbot.stream()
# returns streamed AI chunks, not only AIMessage.
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,   # <-- CHANGED
    HumanMessage,
    ToolMessage
)
# ============================================

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
 
 
# =========================== Utilities =========================== 
def generate_thread_id(): 
    # Keep thread ids as strings everywhere (the DB returns strings too).
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


# ================= CHANGE 2 =================
# Added this function because Gemini can return
# message.content in this format:
#
# [
#     {
#         "type": "text",
#         "text": "Hello..."
#     }
# ]
#
# We only want to display "Hello..."
def extract_text(content):
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
# ============================================


# ======================= Session Initialization =================== 
if "message_history" not in st.session_state: 
    st.session_state["message_history"] = [] 
 
if "thread_id" not in st.session_state: 
    st.session_state["thread_id"] = generate_thread_id() 
 
if "chat_threads" not in st.session_state:
    # On a public demo, visitors should not see each other's chats, so each
    # browser session starts with an empty list. Locally, show all saved chats.
    if os.getenv("PUBLIC_MODE", "false").lower() == "true":
        st.session_state["chat_threads"] = []
    else:
        st.session_state["chat_threads"] = retrieve_all_threads()
 
if "ingested_docs" not in st.session_state: 
    st.session_state["ingested_docs"] = {} 

# How many messages a visitor can send with the app owner's Gemini key.
# Visitors who paste their own key have no limit.
FREE_MESSAGE_LIMIT = int(os.getenv("FREE_MESSAGE_LIMIT", "10"))

if "free_messages_used" not in st.session_state:
    st.session_state["free_messages_used"] = 0
 
add_thread(st.session_state["thread_id"]) 
 
thread_key = str(st.session_state["thread_id"]) 
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {}) 
threads = st.session_state["chat_threads"][::-1] 
selected_thread = None 
 
# ============================ Sidebar ============================ 
st.sidebar.title("LangGraph PDF Chatbot") 
st.sidebar.markdown(f"**Current chat:** {get_chat_title(thread_key)}")
 
if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat() 
    st.rerun() 

with st.sidebar.expander("🔑 Use your own Gemini key"):
    user_api_key = st.text_input(
        "Gemini API key (optional)",
        type="password",
        help="Get a free key at aistudio.google.com. It is only used for this session and never saved.",
    ).strip()

if user_api_key:
    st.sidebar.caption("Using your own key — no message limit.")
else:
    left = FREE_MESSAGE_LIMIT - st.session_state["free_messages_used"]
    st.sidebar.caption(f"Free messages left: {max(left, 0)} / {FREE_MESSAGE_LIMIT}")

# Use the visitor's key if they gave one, otherwise the default chatbot
active_chatbot = build_chatbot(user_api_key) if user_api_key else chatbot
 
if thread_docs: 
    latest_doc = list(thread_docs.values())[-1] 
    st.sidebar.success( 
        f"Using `{latest_doc.get('filename')}` " 
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)" 
    ) 
else: 
    st.sidebar.info("No PDF indexed yet.") 
 
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
    # Per-thread key so "New Chat" starts with an empty uploader instead of
    # silently re-indexing the previous chat's PDF into the new thread.
    key=f"uploader-{thread_key}",
) 

if uploaded_pdf: 
    if uploaded_pdf.name in thread_docs: 
        st.sidebar.info(
            f"`{uploaded_pdf.name}` already processed for this chat."
        ) 
    else: 
        with st.sidebar.status(
            "Indexing PDF…",
            expanded=True
        ) as status_box: 

            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                    api_key=user_api_key or None,
                )

                thread_docs[uploaded_pdf.name] = summary

                status_box.update(
                    label="✅ PDF indexed",
                    state="complete",
                    expanded=False
                )
            except Exception as e:
                # e.g. a scanned PDF with no text, or an API limit
                status_box.update(
                    label=f"❌ Could not index PDF: {e}",
                    state="error",
                    expanded=True
                )
 
st.sidebar.subheader("Past conversations") 

if not threads: 
    st.sidebar.write("No past conversations yet.") 

else: 
    for thread_id in threads: 
        if st.sidebar.button(
            get_chat_title(thread_id),
            key=f"side-thread-{thread_id}",
            use_container_width=True,
            # Highlight the chat that is open right now
            type="primary" if str(thread_id) == thread_key else "secondary",
        ):
            selected_thread = thread_id 
 
# ============================ Main Layout ======================== 
st.title("Multi Utility Chatbot") 
 
# Chat area
for message in st.session_state["message_history"]: 
    with st.chat_message(message["role"]): 

        # ================= CHANGE 3 =================
        # Changed st.text() -> st.markdown()
        #
        # st.text() prints markdown symbols literally.
        # st.markdown() properly renders AI responses.
        st.markdown(message["content"])
        # ============================================
 
user_input = st.chat_input(
    "Ask about your document or use tools"
) 

# Stop visitors from using up the owner's free Gemini quota
if user_input and not user_api_key and st.session_state["free_messages_used"] >= FREE_MESSAGE_LIMIT:
    st.warning(
        f"You have used all {FREE_MESSAGE_LIMIT} free messages. "
        "Add your own Gemini key in the sidebar (🔑) to keep chatting."
    )
    user_input = None
 
if user_input: 

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    ) 

    with st.chat_message("user"): 

        # ================= CHANGE 4 =================
        # Changed st.text() -> st.markdown()
        st.markdown(user_input)
        # ============================================
 
    CONFIG = { 
        "configurable": {"thread_id": thread_key}, 
        "metadata": {"thread_id": thread_key}, 
        "run_name": "chat_turn", 
    } 
 
    with st.chat_message("assistant"): 
        status_holder = {"box": None} 
 
        def ai_only_stream():
            try:
                yield from stream_answer()
            except Exception as e:
                # Show a friendly message instead of a red crash screen
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    yield "⚠️ Too many requests right now (Gemini free-tier limit). Please wait a minute and try again."
                elif "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                    yield "⚠️ This Gemini API key is not valid. Please check the key in the sidebar."
                else:
                    yield f"⚠️ Something went wrong: {e}"

        def stream_answer():

            for message_chunk, _ in active_chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):

                # Tool messages are still handled exactly
                # as they were in your original code.
                if isinstance(message_chunk, ToolMessage): 

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    ) 

                    if status_holder["box"] is None: 

                        status_holder["box"] = st.status( 
                            f"🔧 Using `{tool_name}` …",
                            expanded=True
                        ) 

                    else: 

                        status_holder["box"].update( 
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        ) 


                # ================= CHANGE 5 =================
                # OLD:
                #
                # if isinstance(message_chunk, AIMessage):
                #     yield message_chunk.content
                #
                # NEW:
                #
                # stream_mode="messages" gives us
                # AIMessageChunk objects.
                #
                # We extract only the actual text from
                # message_chunk.content.
                if isinstance(message_chunk, AIMessageChunk):

                    text = extract_text(
                        message_chunk.content
                    )

                    if text:
                        yield text
                # ============================================


        ai_message = st.write_stream(
            ai_only_stream()
        ) 
 
        if status_holder["box"] is not None: 

            status_holder["box"].update( 
                label="✅ Tool finished",
                state="complete",
                expanded=False
            ) 
 
    st.session_state["message_history"].append( 
        {
            "role": "assistant",
            "content": ai_message
        }
    )

    # Count only real answers, not error messages (they start with ⚠️)
    if not user_api_key and not str(ai_message).startswith("⚠️"):
        st.session_state["free_messages_used"] += 1

    # First message of a brand-new chat (1 user + 1 assistant message so far):
    # give the chat a title and rerun so the sidebar shows it right away.
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

    messages = load_conversation(
        selected_thread
    ) 
 
    temp_messages = [] 

    for msg in messages: 

        # Skip tool outputs (raw JSON) and tool-call-only AI messages,
        # otherwise they show up as junk/empty assistant bubbles.
        if not isinstance(msg, (HumanMessage, AIMessage)):
            continue
        if not extract_text(msg.content).strip():
            continue

        role = (
            "user"
            if isinstance(msg, HumanMessage)
            else "assistant"
        ) 

        # ================= CHANGE 6 =================
        # OLD:
        #
        # "content": msg.content
        #
        # NEW:
        #
        # Use extract_text() so old conversations
        # don't display:
        #
        # [{"type": "text", "text": "..."}]
        temp_messages.append(
            {
                "role": role,
                "content": extract_text(msg.content)
            }
        )
        # ============================================
 
    st.session_state["message_history"] = temp_messages 

    st.session_state["ingested_docs"].setdefault(
        str(selected_thread),
        {}
    ) 

    st.rerun()