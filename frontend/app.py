"""Streamlit chat UI for the RAG Document Assistant."""

import streamlit as st

from api_client import API_BASE_URL, APIError, ask, health

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="centered")

# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("⚙️ Backend")
    st.caption(f"API_BASE_URL: `{API_BASE_URL or 'not set'}`")

    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=4)

    if st.button("Check connection", use_container_width=True):
        try:
            info = health()
            st.success(f"Status: {info['status']}")
            st.json(info)
        except APIError as exc:
            st.error(str(exc))

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Answers are generated only from the indexed documents. "
        "If the answer is not in them, the assistant says so instead of guessing."
    )

# ------------------------------------------------------------------- main ----
st.title("📚 RAG Document Assistant")
st.caption("Ask a question — get an answer grounded in your document collection, with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(payload: dict) -> None:
    contexts = payload.get("contexts", [])
    if not contexts:
        return
    with st.expander(f"📎 Sources ({len(contexts)})", expanded=False):
        for i, ctx in enumerate(contexts, start=1):
            st.markdown(f"**[{i}] {ctx['citation']}** · similarity `{ctx['score']:.3f}`")
            st.caption(ctx["preview"] + "…")
            if i < len(contexts):
                st.divider()


# replay history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("payload"):
            render_sources(message["payload"])
            meta = message["payload"]
            st.caption(f"⏱ {meta['latency_ms']} ms · model `{meta['model']}`")

question = st.chat_input("Ask a question about the documents…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context and generating a grounded answer…"):
                payload = ask(question, top_k=top_k)
        except APIError as exc:
            st.error(f"❌ {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ {exc}", "payload": None}
            )
        else:
            if not payload["grounded"]:
                st.warning("No sufficiently relevant passage was found in the documents.")
            st.markdown(payload["answer"])
            render_sources(payload)
            st.caption(f"⏱ {payload['latency_ms']} ms · model `{payload['model']}`")
            st.session_state.messages.append(
                {"role": "assistant", "content": payload["answer"], "payload": payload}
            )
