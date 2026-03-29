from pydantic import BaseModel
from typing import Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "tinyllama"
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    stream: bool = True

class ModelInfo(BaseModel):
    id: str
    name: str
    path: str
    size_mb: int