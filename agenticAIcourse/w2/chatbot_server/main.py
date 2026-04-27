import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI

from models import (
    MessageRecord,
    MessageRequest,
    MessageResponse,
    SessionCreate,
    SessionDetail,
    SessionResponse,
)

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)
model = os.environ["LLM_MODEL"]

app = FastAPI(title="Chatbot Server")


@dataclass
class Session:
    session_id: str
    created_at: str
    system_prompt: str | None
    messages: list[dict] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


sessions: dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesiunea nu există")
    return sessions[session_id]


@app.post("/sessions", response_model=SessionResponse)
def create_session(body: SessionCreate):
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    sessions[session_id] = Session(
        session_id=session_id,
        created_at=created_at,
        system_prompt=body.system_prompt,
    )
    return SessionResponse(session_id=session_id, created_at=created_at)


@app.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session_detail(session_id: str):
    session = get_session(session_id)
    return SessionDetail(
        session_id=session.session_id,
        created_at=session.created_at,
        messages=[MessageRecord(**m) for m in session.messages],
        total_prompt_tokens=session.total_prompt_tokens,
        total_completion_tokens=session.total_completion_tokens,
    )


@app.post("/sessions/{session_id}/messages", response_model=MessageResponse)
def send_message(session_id: str, body: MessageRequest):
    session = get_session(session_id)

    session.messages.append({"role": "user", "content": body.content})

    history = []
    if session.system_prompt:
        history.append({"role": "system", "content": session.system_prompt})
    history.extend(session.messages)

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=history,
    )

    reply = response.choices[0].message.content
    session.messages.append({"role": "assistant", "content": reply})

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    session.total_prompt_tokens += prompt_tokens
    session.total_completion_tokens += completion_tokens

    return MessageResponse(
        content=reply,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    get_session(session_id)
    del sessions[session_id]
    return {"detail": "Sesiunea a fost ștearsă"}
