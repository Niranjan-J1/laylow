import json
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from api.models import ChatRequest, ModelInfo
from api import bridge
from api import rag
from api import memory

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

@app.get("/memory")
def get_memory():
    return {"facts": memory.get_facts()}

@app.delete("/memory")
def clear_memory():
    memory.clear_facts()
    return {"status": "cleared"}

@app.get("/conversations")
def get_conversations():
    return memory.get_conversations()

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if not bridge.list_models():
        raise HTTPException(status_code=404,
            detail="No models found. Add a .gguf file to the models/ folder.")

    # Extract and save any facts from the latest user message
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), None
    )
    if last_user:
        new_facts = memory.extract_facts_from_message(last_user)
        for fact in new_facts:
            memory.add_fact(fact)

    # Build system prompt with memory + RAG context
    system_parts = []

    # Add memory facts
    facts = memory.get_facts()
    if facts:
        facts_text = "\n".join(f"- {f}" for f in facts[-10:])
        system_parts.append(f"Things you know about the user:\n{facts_text}")

    # Add RAG context
    rag_chunks = []
    if last_user:
        rag_chunks = rag.retrieve(last_user, top_k=3)
        if rag_chunks:
            context_text = "\n\n".join(rag_chunks)
            system_parts.append(f"Relevant context from uploaded documents:\n{context_text}")

    # Assemble augmented messages
    augmented = list(messages)
    if system_parts:
        system_content = "You are a helpful private AI assistant running locally.\n\n"
        system_content += "\n\n".join(system_parts)
        augmented = [{"role": "system", "content": system_content}] + augmented

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

            if rag_chunks:
                yield {
                    "event": "context",
                    "data": json.dumps({"chunks_used": len(rag_chunks)})
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

            # Save conversation to history
            full_response = "".join(tokens)
            all_messages = messages + [{"role": "assistant", "content": full_response}]
            memory.save_conversation(all_messages)

        except FileNotFoundError as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_stream())

@app.get("/")
def root():
    return {"name": "laylow", "version": "0.1.0", "docs": "/docs"}