"""Streamlit interface for SuccessGuideAI.

Run with:  streamlit run app.py
The OpenAI API key is read from the OPENAI_API_KEY environment variable,
from .streamlit/secrets.toml, or typed into the sidebar. It is never stored in the code.
"""

import os
from pathlib import Path

import streamlit as st

from guide import Assistant, CourseLibrary

st.set_page_config(page_title="SuccessGuideAI", page_icon="🎓", layout="centered")
DEFAULT_MODEL = os.getenv("SUCCESSGUIDE_MODEL", "gpt-4o-mini")


@st.cache_resource
def get_library() -> CourseLibrary:
    return CourseLibrary("data/chroma")


def get_api_key() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    secrets_files = [Path(".streamlit/secrets.toml"), Path.home() / ".streamlit/secrets.toml"]
    if any(p.exists() for p in secrets_files):  # avoid Streamlit's "no secrets file" warning
        return st.secrets.get("OPENAI_API_KEY", "")
    return ""


library = get_library()

# ---- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("Course materials")
    files = st.file_uploader("Syllabi, learning objectives or notes (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    if files and st.button("Add to library"):
        with st.spinner("Indexing…"):
            n = sum(library.add_file(f.name, f.read()) for f in files)
        st.success(f"Added {n} passages.")
    if st.button("Load sample materials"):
        st.success(f"Added {library.add_folder('sample_materials')} passages from the sample course.")
    st.caption(f"Library: {library.count()} passages from {len(library.sources())} file(s)")

    st.header("Settings")
    api_key = get_api_key() or st.text_input("OpenAI API key", type="password", help="Used only for this session.")
    model = st.text_input("Model", DEFAULT_MODEL)
    if st.button("New conversation"):
        st.session_state.pop("assistant", None)
        st.session_state.pop("messages", None)

# ---- main -------------------------------------------------------------------
st.title("SuccessGuideAI")
st.write("Ask how to succeed in your courses or which careers fit your interests. Answers cite your own course materials.")
interests = st.text_input("Your interests", placeholder="e.g. programming, design, teaching")

if not api_key:
    st.info("Add an OpenAI API key in the sidebar (or set OPENAI_API_KEY) to start chatting.")
    st.stop()

if "assistant" not in st.session_state:
    from langchain_openai import ChatOpenAI

    st.session_state.assistant = Assistant(library, ChatOpenAI(model=model, temperature=0.2, api_key=api_key))
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("e.g. How can I do well in this course? What careers fit my interests?"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = st.session_state.assistant.ask(question, interests)
        st.markdown(answer.text)
        if answer.sources:
            with st.expander("Sources from your materials"):
                for i, p in enumerate(answer.sources, start=1):
                    st.markdown(f"**[{i}] {p.source}**: {p.text[:300]}…")
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer.text},
    ]
