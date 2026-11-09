"""
Streamlit Application Layer for RAG Knowledge Assistant.

Orchestrates document ingestion, semantic vector retrieval, and grounded
answer generation using the existing modular backend pipeline.
"""

import importlib
import os
import tempfile
from typing import Any, Dict, List, Optional

import streamlit as st

import src.rag_generator
importlib.reload(src.rag_generator)
from src.database import compute_content_hash, get_db_service
from src.db_config import is_persistence_enabled, mask_database_url
from src.embedding_generator import EmbeddingGenerator
from src.ingestion_pipeline import build_vector_store
from src.llm_factory import get_llm_provider_instance
from src.rag_generator import RAGGenerator
from src.retriever import Retriever
from src.storage_config import get_storage_provider_name
from src.storage_factory import create_storage_provider
from src.vector_store import VectorStore

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "llama3.2:latest"
CALIBRATED_DISTANCE_THRESHOLD = 1.60
TOP_K_RETRIEVAL = 3


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_generator() -> EmbeddingGenerator:
    """
    Cache the SentenceTransformer embedding model in memory across reruns.
    Ensures the model weights are loaded only once per application lifecycle.
    """
    return EmbeddingGenerator(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading neural reranker...")
def get_neural_reranker() -> Optional[Any]:
    """
    Cache the CrossEncoder neural reranker model in memory across reruns.
    Gracefully returns None if sentence_transformers CrossEncoder is unavailable.
    """
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL_NAME)
    except Exception:
        return None


def init_session_state() -> None:
    """Initialize Streamlit session state defaults safely."""
    defaults = {
        "embedding_generator": None,
        "vector_store": None,
        "retriever": None,
        "rag_generator": None,
        "file_identifier": None,
        "uploaded_file_name": None,
        "storage_key": None,
        "num_chunks": 0,
        "messages": [],
        "active_document_id": None,
        "active_conversation_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def clear_document_state() -> None:
    """Clear document-specific state and chat history while preserving global resources."""
    st.session_state["vector_store"] = None
    st.session_state["retriever"] = None
    st.session_state["rag_generator"] = None
    st.session_state["file_identifier"] = None
    st.session_state["uploaded_file_name"] = None
    st.session_state["storage_key"] = None
    st.session_state["num_chunks"] = 0
    st.session_state["messages"] = []
    st.session_state["active_document_id"] = None
    st.session_state["active_conversation_id"] = None


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """
    Render source citations in a collapsible expander below an assistant answer.
    Displays source filename, page number, and semantic retrieval distance.
    """
    if not sources or len(sources) == 0:
        return

    with st.expander(f"View Sources ({len(sources)})", expanded=False):
        for idx, src in enumerate(sources, start=1):
            source_file = src.get("source", "Unknown")
            page_num = src.get("page_number", "Unknown")
            dist = src.get("distance")
            dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else str(dist)

            st.markdown(
                f"- **Source {idx}:** `{source_file}` | "
                f"**Page:** {page_num} | "
                f"**Distance:** `{dist_str}`"
            )


def get_recent_chat_history(max_turns: int = 2) -> List[Dict[str, str]]:
    """
    Extract the last `max_turns` completed (user, assistant) turns from
    st.session_state["messages"], stripping UI-only metadata and error messages.
    """
    raw_messages = st.session_state.get("messages", [])
    if not raw_messages:
        return []

    completed_turns: List[Dict[str, str]] = []
    idx = 0
    while idx < len(raw_messages):
        msg = raw_messages[idx]
        if msg.get("role") == "user":
            user_content = msg.get("content", "").strip()
            if idx + 1 < len(raw_messages) and raw_messages[idx + 1].get("role") == "assistant":
                asst_content = raw_messages[idx + 1].get("content", "").strip()
                if (
                    not asst_content.startswith("Error generating answer")
                    and not asst_content.startswith("Unable to reach")
                ):
                    completed_turns.append({"role": "user", "content": user_content})
                    completed_turns.append({"role": "assistant", "content": asst_content})
                idx += 2
                continue
        idx += 1

    max_messages = max_turns * 2
    return completed_turns[-max_messages:]


def ingest_uploaded_pdf(uploaded_file: Any) -> None:
    """
    Process an uploaded PDF file through object storage,
    the ingestion pipeline, and initialize Retriever and RAGGenerator.
    """
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    # Only process if this is a newly uploaded or changed document
    if st.session_state.get("file_identifier") == file_id:
        return

    # Invalidate old document state and conversation history
    clear_document_state()

    with st.spinner(f"Processing '{uploaded_file.name}' and building vector index..."):
        storage_provider = None
        storage_key = None
        try:
            # 1. Obtain configured object storage provider
            storage_provider = create_storage_provider()
            file_bytes = uploaded_file.getvalue()

            # 2. Store binary object via storage abstraction
            storage_key = storage_provider.upload(file_bytes, filename=uploaded_file.name)

            # 3. Ingestion pipeline: write to temporary working directory for loading & chunking
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(file_bytes)

                # Retrieve cached shared embedding generator
                emb_gen = get_embedding_generator()
                st.session_state["embedding_generator"] = emb_gen

                # Ingestion: load -> clean -> chunk -> embed -> index
                vector_store: VectorStore = build_vector_store(
                    temp_pdf_path,
                    embedding_generator=emb_gen,
                )

                # Initialize retriever
                retriever = Retriever(
                    vector_store=vector_store,
                    embedding_generator=emb_gen,
                )

                # Initialize configured LLM provider and RAG generator with calibrated production parameters
                llm_provider = get_llm_provider_instance()
                reranker_instance = get_neural_reranker()
                rag_generator = RAGGenerator(
                    retriever=retriever,
                    model_name=llm_provider.model_name,
                    default_max_distance=CALIBRATED_DISTANCE_THRESHOLD,
                    llm_provider=llm_provider,
                    reranker=reranker_instance,
                )

                # Store active document components in session state
                st.session_state["vector_store"] = vector_store
                st.session_state["retriever"] = retriever
                st.session_state["rag_generator"] = rag_generator
                st.session_state["file_identifier"] = file_id
                st.session_state["uploaded_file_name"] = uploaded_file.name
                st.session_state["storage_key"] = storage_key
                st.session_state["num_chunks"] = len(vector_store.documents)
                st.session_state["messages"] = []

                # Optional PostgreSQL relational persistence
                if is_persistence_enabled():
                    try:
                        db = get_db_service()
                        db.initialize()
                        c_hash = compute_content_hash(file_bytes)
                        chunk_count = len(vector_store.documents) if hasattr(vector_store, "documents") else 0
                        doc_record = db.documents.create_or_get_document(
                            filename=uploaded_file.name,
                            content_hash=c_hash,
                            file_size_bytes=uploaded_file.size,
                            storage_path=storage_key,
                            chunk_count=chunk_count,
                            storage_key=storage_key,
                        )
                        st.session_state["active_document_id"] = doc_record["id"]
                        conv_record = db.conversations.create_conversation(
                            document_id=doc_record["id"],
                            title=f"Chat: {uploaded_file.name}",
                        )
                        st.session_state["active_conversation_id"] = conv_record["id"]
                    except Exception as db_err:
                        safe_msg = mask_database_url(str(db_err))
                        st.error(f"PostgreSQL persistence error: {safe_msg}")

        except Exception as err:
            if storage_provider and storage_key:
                try:
                    storage_provider.delete(storage_key)
                except Exception:
                    pass
            clear_document_state()
            st.error(f"Failed to process document: {err}")


def main() -> None:
    st.set_page_config(
        page_title="RAG Knowledge Assistant",
        page_icon="📄",
        layout="wide",
    )

    init_session_state()

    # Sidebar: Document Upload and System Information
    with st.sidebar:
        st.header("Document Management")
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload a PDF file to index for contextual question-answering.",
        )

        if uploaded_file is None:
            if st.session_state.get("file_identifier") is not None:
                clear_document_state()
            st.info("No document loaded")
        else:
            ingest_uploaded_pdf(uploaded_file)

        st.divider()
        st.subheader("System Status")

        is_ready = st.session_state.get("rag_generator") is not None

        if is_ready:
            st.success("Ready for Queries")
            st.markdown(f"**Document:** `{st.session_state.get('uploaded_file_name')}`")
            st.markdown(f"**Indexed Chunks:** `{st.session_state.get('num_chunks')}`")
        else:
            st.warning("Awaiting Document")

        st.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`")
        st.markdown(f"**Neural Reranker:** `{RERANKER_MODEL_NAME}`")
        st.markdown(f"**Vector Store:** `FAISS IndexFlatL2`")
        st.markdown(f"**Storage Provider:** `{get_storage_provider_name()}`")
        st.markdown(f"**LLM Generator:** `{LLM_MODEL_NAME}`")
        st.markdown(f"**Relevance Cutoff:** `{CALIBRATED_DISTANCE_THRESHOLD:.2f}`")

        # Persistent Conversations Management (Active only when PERSISTENCE_ENABLED=true)
        if is_persistence_enabled() and st.session_state.get("active_document_id"):
            st.divider()
            st.subheader("Saved Conversations")
            try:
                db = get_db_service()
                conv_list = db.conversations.list_by_document(st.session_state["active_document_id"])
                if conv_list:
                    conv_map = {c["id"]: c["title"] for c in conv_list}
                    curr_conv_id = st.session_state.get("active_conversation_id")
                    options = list(conv_map.keys())
                    curr_idx = options.index(curr_conv_id) if curr_conv_id in options else 0
                    chosen_conv_id = st.selectbox(
                        "Active Conversation",
                        options=options,
                        format_func=lambda cid: conv_map.get(cid, cid),
                        index=curr_idx,
                    )
                    if chosen_conv_id != curr_conv_id:
                        st.session_state["active_conversation_id"] = chosen_conv_id
                        stored_msgs = db.messages.list_by_conversation(chosen_conv_id)
                        st.session_state["messages"] = [
                            {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
                            for m in stored_msgs
                        ]
                        st.rerun()

                if st.button("+ New Conversation"):
                    new_conv = db.conversations.create_conversation(
                        document_id=st.session_state["active_document_id"],
                        title=f"Chat {len(conv_list) + 1}: {st.session_state.get('uploaded_file_name')}",
                    )
                    st.session_state["active_conversation_id"] = new_conv["id"]
                    st.session_state["messages"] = []
                    st.rerun()
            except Exception as db_err:
                st.error(f"Failed to load conversations: {mask_database_url(str(db_err))}")

    # Main Area: Title, State, and Chat Interface
    st.title("Context-Aware AI Knowledge Assistant")
    st.caption("Grounded document question-answering powered by local semantic search and Ollama.")

    # Readiness State Banner
    if not is_ready:
        st.info("👈 Please upload a PDF document in the sidebar to begin asking questions.")
    else:
        st.divider()

        # Render full conversation history for the current session
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                msg_sources = msg.get("sources", [])
                if msg_sources and len(msg_sources) > 0:
                    render_sources(msg_sources)

        # Chat Input Area
        prompt = st.chat_input("Ask a question about your document...")

        if prompt:
            # Extract bounded recent completed chat history (last 2 completed Q&A turns)
            recent_history = get_recent_chat_history(max_turns=2)

            # Append and render user message
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Persist user message if persistence is enabled
            if is_persistence_enabled() and st.session_state.get("active_conversation_id"):
                try:
                    db = get_db_service()
                    db.messages.create_message(
                        conversation_id=st.session_state["active_conversation_id"],
                        role="user",
                        content=prompt,
                    )
                except Exception as db_err:
                    st.warning(f"Failed to persist user message: {mask_database_url(str(db_err))}")

            # Query RAG pipeline and render assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching document and generating answer..."):
                    try:
                        rag_gen: RAGGenerator = st.session_state["rag_generator"]
                        result = rag_gen.generate_answer(
                            prompt,
                            top_k=TOP_K_RETRIEVAL,
                            chat_history=recent_history,
                        )
                        answer = result.get("answer", "")
                        sources = result.get("sources", [])

                        # Defensive source pruning: ensure unavailable answers have zero sources
                        if rag_gen._is_unavailable_answer(answer):
                            sources = []

                        st.markdown(answer)
                        if sources and len(sources) > 0:
                            render_sources(sources)

                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                        # Persist assistant message if persistence is enabled
                        if is_persistence_enabled() and st.session_state.get("active_conversation_id"):
                            try:
                                db = get_db_service()
                                db.messages.create_message(
                                    conversation_id=st.session_state["active_conversation_id"],
                                    role="assistant",
                                    content=answer,
                                    sources=sources,
                                )
                            except Exception as db_err:
                                st.warning(f"Failed to persist assistant message: {mask_database_url(str(db_err))}")

                    except Exception as err:
                        err_str = str(err)
                        if "connection" in err_str.lower() or "connect" in err_str.lower():
                            error_text = (
                                f"Unable to reach the Ollama service: {err_str}\n\n"
                                "Please ensure Ollama is running (`ollama serve`) and model "
                                f"`{LLM_MODEL_NAME}` is available."
                            )
                        else:
                            error_text = f"Error generating answer: {err_str}"

                        st.error(error_text)
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": error_text,
                            "sources": [],
                        })


if __name__ == "__main__":
    main()
