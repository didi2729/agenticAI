import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, requests
from openai import OpenAI

from models import (
    CreateSessionRequest,
    MessageRequest,
    MessageResponse,
    SessionResponse,
    SessionHistoryResponse,
    Message,
)

load_dotenv()

sessions: dict[str, dict] = {}
llm_client: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_client["client"] = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )
    yield
    llm_client.clear()


app = FastAPI(lifespan=lifespan)


def creeaza_sesiune_autentificata(token: str) -> requests.Session:
    """Creează o sesiune auth"""
    sesiune = requests.Session()
    sesiune.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    return sesiune

 

def _make_session(system_prompt: str | None) -> dict:
    session_id = str(uuid4())
    sessions[session_id] = {
        "id": session_id,
        "messages": [],
        "system_prompt": system_prompt,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return sessions[session_id]


@app.post("/sessions")
def create_session(body: CreateSessionRequest) -> SessionResponse:
    session = _make_session(body.system_prompt)
    return SessionResponse(id=session["id"], created_at=session["created_at"])


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> SessionHistoryResponse:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesiunea nu există")
    return SessionHistoryResponse(
        id=session["id"],
        system_prompt=session["system_prompt"],
        messages=[Message(**m) for m in session["messages"]]
    )


@app.post("/sessions/{session_id}/messages")
def send_message(session_id: str, body: MessageRequest) -> MessageResponse:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesiunea nu există")

    session["messages"].append({"role": "user", "content": body.content})

    history = []
    if session["system_prompt"]:
        history.append({"role": "system", "content": session["system_prompt"]})
    history.extend(session["messages"])

    client = llm_client["client"]
    model = os.environ["LLM_MODEL"]

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=history,
    )
    reply = response.choices[0].message.content

    session["messages"].append({"role": "assistant", "content": reply})

    return MessageResponse(reply=reply)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesiunea nu există")
    del sessions[session_id]
    return {"detail": f"Sesiunea {session_id} a fost ștearsă"}

