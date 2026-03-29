import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from api.models import ChatRequest, ModelInfo
from api import bridge

app = FastAPI(title="laylow API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "engine": "llama-cpp-python"}

@app.get("/models", response_model=list[ModelInfo])
def get_models():
    return bridge.list_models()

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if not bridge.list_models():
        raise HTTPException(status_code=404,
                            detail="No models found. Add a .gguf file to the models/ folder.")

    async def event_stream():
        try:
            loop = asyncio.get_event_loop()

            def run_generation():
                tokens = []
                for token in bridge.generate_stream(
                    model_id=req.model,
                    messages=messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p
                ):
                    tokens.append(token)
                return tokens

            tokens = await loop.run_in_executor(None, run_generation)

            for token in tokens:
                yield {
                    "event": "token",
                    "data": json.dumps({"token": token})
                }

            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"})
            }

        except FileNotFoundError as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_stream())

@app.get("/")
def root():
    return {
        "name": "laylow",
        "version": "0.1.0",
        "docs": "/docs"
    }