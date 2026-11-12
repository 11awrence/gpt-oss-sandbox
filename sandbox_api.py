from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
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
DEFAULT_SYSTEM = os.getenv("SYSTEM_PROMPT", "You are a helpful local assistant with limited tools.")
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", "20"))

app = FastAPI(title="Sandbox API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory cache, loaded from SQLite on startup
_conversations: dict[str, dict] = {}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def create_title(message: str, max_length: int = 60) -> str:
    normalized = " ".join(message.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."

@app.on_event("startup")
async def startup():
    initialize_database()
    for conv in load_all_conversations():
        _conversations[conv["conversation_id"]] = conv

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    messages: list[dict]

class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str

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

    payload: dict[str, Any] = {
        "messages": history,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(f"{LLAMA_URL}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    history.append({"role": "assistant", "content": answer})

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

    # simple limit
    if len(_conversations) > MAX_CONVERSATIONS:
        oldest = sorted(_conversations.values(), key=lambda c: c["updated_at"])[0]
        _conversations.pop(oldest["conversation_id"], None)
        db_delete_conversation(oldest["conversation_id"])

    return ChatResponse(
        response=answer,
        conversation_id=conversation_id,
        messages=history,
    )

@app.get("/v1/conversations", response_model=list[ConversationSummary])
def list_conversations():
    convs = sorted(_conversations.values(), key=lambda c: c["updated_at"], reverse=True)
    return [
        ConversationSummary(
            conversation_id=c["conversation_id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
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
