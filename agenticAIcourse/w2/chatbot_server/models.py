from pydantic import BaseModel
from typing import Optional


class SessionCreate(BaseModel):
    system_prompt: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    created_at: str


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int


class MessageRecord(BaseModel):
    role: str
    content: str


class SessionDetail(BaseModel):
    session_id: str
    created_at: str
    messages: list[MessageRecord]
    total_prompt_tokens: int
    total_completion_tokens: int
