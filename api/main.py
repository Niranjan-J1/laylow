import json
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from api.models import ChatRequest, ModelInfo
from api import bridge
from api import rag

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

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        result = rag.add_document(contents, file.filename)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/documents")
def get_documents():
    return rag.list_documents()

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if not bridge.list_models():
        raise HTTPException(status_code=404,
            detail="No models found. Add a .gguf file to the models/ folder.")

    # RAG: retrieve relevant context for the last user message
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), None
    )

    rag_context = []
    if last_user:
        rag_context = rag.retrieve(last_user, top_k=3)

    # Inject context into the system message if we found anything
    augmented = list(messages)
    if rag_context:
        context_text = "\n\n".join(rag_context)
        system_msg = {
            "role": "system",
            "content": f"Use the following context to help answer the user's question. "
                       f"If the context is not relevant, answer from your own knowledge.\n\n"
                       f"Context:\n{context_text}"
        }
        augmented = [system_msg] + augmented

    async def event_stream():
        try:
            loop = asyncio.get_event_loop()

            def run_generation():
                tokens = []
                for token in bridge.generate_stream(
                    model_id=req.model,
                    messages=augmented,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p
                ):
                    tokens.append(token)
                return tokens

            tokens = await loop.run_in_executor(None, run_generation)

            # Send RAG source info first if we used context
            if rag_context:
                yield {
                    "event": "context",
                    "data": json.dumps({"chunks_used": len(rag_context)})
                }

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
    return {"name": "laylow", "version": "0.1.0", "docs": "/docs"}