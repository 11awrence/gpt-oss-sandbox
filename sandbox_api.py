from __future__ import annotations

import os
import uuid
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import (
    initialize_database,
    save_conversation,
    load_all_conversations,
    delete_conversation as db_delete_conversation,
)

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080")
DEFAULT_SYSTEM = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful local assistant. You have tools to list, read, and write files, and run commands, but only inside the sandbox directory. No internet. If you need a Python package, ask the user to install it.",
)
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", "20"))
SANDBOX_DIR = Path(os.getenv("SANDBOX_DIR", str(Path.home() / "sandbox"))).resolve()
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sandbox API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_conversations: dict[str, dict] = {}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def create_title(message: str, max_length: int = 60) -> str:
    normalized = " ".join(message.split())
    return normalized if len(normalized) <= max_length else normalized[: max_length - 3].rstrip() + "..."

def safe_path(user_path: str) -> Path:
    """Resolve path and ensure it stays inside SANDBOX_DIR."""
    p = (SANDBOX_DIR / user_path).resolve()
    if not str(p).startswith(str(SANDBOX_DIR)):
        raise ValueError("Path escapes sandbox")
    return p

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories inside the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside sandbox (default '.')"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside sandbox"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a text file inside the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command inside the sandbox directory. No network.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run"}
                },
                "required": ["command"],
            },
        },
    },
]

def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "list_files":
            path = safe_path(args.get("path", "."))
            if not path.exists():
                return "Path does not exist"
            items = [p.name + ("/" if p.is_dir() else "") for p in sorted(path.iterdir())]
            return "\n".join(items) or "(empty)"

        if name == "read_file":
            path = safe_path(args["path"])
            if not path.is_file():
                return "Not a file or does not exist"
            return path.read_text(errors="replace")[:100_000]

        if name == "write_file":
            path = safe_path(args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"])
            return f"Wrote {path.relative_to(SANDBOX_DIR)}"

        if name == "run_command":
            result = subprocess.run(
                args["command"],
                shell=True,
                cwd=SANDBOX_DIR,
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = result.stdout + result.stderr
            return out[:50_000] or "(no output)"

        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"

@app.on_event("startup")
async def startup():
    initialize_database()
    for conv in load_all_conversations():
        _conversations[conv["conversation_id"]] = conv

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    messages: list[dict]

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{LLAMA_URL}/health")
            r.raise_for_status()
            return {"status": "ok", "llama": r.json(), "conversations": len(_conversations)}
    except Exception as e:
        raise HTTPException(500, f"llama-server unreachable: {e}")

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    conversation_id = req.conversation_id or str(uuid.uuid4())
    now = utc_now()

    existing = _conversations.get(conversation_id)
    if existing:
        history = list(existing["messages"])
        created_at = existing["created_at"]
        title = existing["title"]
    else:
        history = [{"role": "system", "content": DEFAULT_SYSTEM}]
        created_at = now
        title = create_title(req.message)

    history.append({"role": "user", "content": req.message})

    # simple tool loop (max 5 rounds)
    for _ in range(5):
        payload = {
            "messages": history,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "tools": TOOLS,
            "tool_choice": "auto",
        }

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{LLAMA_URL}/v1/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {e}")

        msg = data["choices"][0]["message"]
        history.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            break

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            try:
                args = json.loads(fn["arguments"])
            except Exception:
                args = {}
            result = execute_tool(name, args)
            history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    answer = history[-1].get("content", "") if history[-1].get("role") == "assistant" else ""

    conv = {
        "conversation_id": conversation_id,
        "title": title,
        "messages": history,
        "created_at": created_at,
        "updated_at": now,
    }
    _conversations[conversation_id] = conv

    system_prompt = history[0]["content"] if history and history[0]["role"] == "system" else DEFAULT_SYSTEM
    save_conversation(conversation_id, title, system_prompt, history, created_at, now)

    if len(_conversations) > MAX_CONVERSATIONS:
        oldest = sorted(_conversations.values(), key=lambda c: c["updated_at"])[0]
        _conversations.pop(oldest["conversation_id"], None)
        db_delete_conversation(oldest["conversation_id"])

    return ChatResponse(response=answer, conversation_id=conversation_id, messages=history)

@app.get("/v1/conversations")
def list_conversations():
    convs = sorted(_conversations.values(), key=lambda c: c["updated_at"], reverse=True)
    return [
        {
            "conversation_id": c["conversation_id"],
            "title": c["title"],
            "created_at": c["created_at"],
            "updated_at": c["updated_at"],
        }
        for c in convs
    ]

@app.get("/v1/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conv = _conversations.get(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv

@app.delete("/v1/conversations/{conversation_id}")
def reset_conversation(conversation_id: str):
    cleared = _conversations.pop(conversation_id, None) is not None
    if cleared:
        db_delete_conversation(conversation_id)
    return {"conversation_id": conversation_id, "cleared": cleared}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
