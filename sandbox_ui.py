import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SANDBOX = Path.home() / "sandbox"
SANDBOX.mkdir(exist_ok=True)

st.set_page_config(page_title="Sandbox", page_icon="🖤", layout="wide")

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "backend" not in st.session_state:
        st.session_state.backend = BACKEND

init_state()

def api(path: str) -> str:
    return st.session_state.backend.rstrip("/") + path

def health() -> tuple[bool, str]:
    try:
        r = httpx.get(api("/health"), timeout=5)
        r.raise_for_status()
        return True, "connected"
    except Exception as e:
        return False, str(e)

def list_convs() -> list[dict]:
    try:
        r = httpx.get(api("/v1/conversations"), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def load_conv(cid: str):
    r = httpx.get(api(f"/v1/conversations/{cid}"), timeout=10)
    r.raise_for_status()
    data = r.json()
    st.session_state.conversation_id = cid
    st.session_state.messages = [
        m for m in data.get("messages", [])
        if m.get("role") in ("user", "assistant")
    ]

def delete_conv(cid: str):
    httpx.delete(api(f"/v1/conversations/{cid}"), timeout=10)
    if st.session_state.conversation_id == cid:
        st.session_state.messages = []
        st.session_state.conversation_id = None

def new_chat():
    st.session_state.messages = []
    st.session_state.conversation_id = None

def send(message: str) -> str:
    payload: dict[str, Any] = {"message": message}
    if st.session_state.conversation_id:
        payload["conversation_id"] = st.session_state.conversation_id
    r = httpx.post(api("/v1/chat"), json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    st.session_state.conversation_id = data["conversation_id"]
    st.session_state.messages = [
        m for m in data.get("messages", [])
        if m.get("role") in ("user", "assistant")
    ]
    return data["response"]

def guess_lang(path: Path) -> str:
    return {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".json": "json", ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
        ".sh": "bash", ".csv": "csv", ".html": "html", ".css": "css", ".sql": "sql",
    }.get(path.suffix.lower(), "text")

st.title("🖤 Sandbox")
st.caption("Local gpt-oss · tools locked to ~/sandbox")

ok, status = health()

with st.sidebar:
    st.header("Chats")
    if st.button("＋ New", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    if ok:
        for c in list_convs():
            cid = c["conversation_id"]
            title = c.get("title") or "Untitled"
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(title, key=f"open_{cid}", use_container_width=True):
                    load_conv(cid)
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{cid}"):
                    delete_conv(cid)
                    st.rerun()
    else:
        st.caption("Backend offline")

    st.divider()
    st.text_input("Backend", key="backend")
    if ok:
        st.success(status)
    else:
        st.error("unreachable")

chat_col, file_col = st.columns([0.65, 0.35])

with chat_col:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

with file_col:
    st.subheader("Sandbox files")
    uploaded = st.file_uploader("Drop a script or data file")
    if uploaded:
        dest = SANDBOX / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        st.success(f"Saved {uploaded.name}")
        st.rerun()

    files = [p for p in sorted(SANDBOX.iterdir()) if p.is_file()]
    if not files:
        st.caption("No files yet")
    for p in files:
        with st.expander(p.name):
            try:
                content = p.read_text(errors="replace")[:8000]
                st.code(content, language=guess_lang(p))
            except Exception:
                st.caption("(binary)")
            st.download_button(
                "Download",
                data=p.read_bytes(),
                file_name=p.name,
                key=f"dl_{p.name}",
            )

# Chat input lives outside the columns so it stays at the bottom
if prompt := st.chat_input("Message the sandbox…", disabled=not ok):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("thinking…"):
        try:
            reply = send(prompt)
        except Exception as e:
            reply = f"Error: {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
