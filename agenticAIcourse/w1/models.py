from pydantic import BaseModel
from typing import Optional


class Message(BaseModel):
    role: str
    content: str


class CreateSessionRequest(BaseModel):
    system_prompt: Optional[str] = None


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    reply: str


class SessionResponse(BaseModel):
    id: str
    created_at: str


class SessionHistoryResponse(BaseModel):
    id: str
    system_prompt: Optional[str] = None
    messages: list[Message] = []
