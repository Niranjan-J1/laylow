import os
import glob
from pathlib import Path
from llama_cpp import Llama

MODELS_DIR = Path(__file__).parent.parent / "models"

_loaded_models: dict[str, Llama] = {}

def get_models_dir() -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR

def list_models() -> list[dict]:
    models = []
    for path in glob.glob(str(MODELS_DIR / "*.gguf")):
        p = Path(path)
        models.append({
            "id": p.stem,
            "name": p.stem,
            "path": str(p),
            "size_mb": int(p.stat().st_size / 1024 / 1024)
        })
    return models

def get_model(model_id: str) -> Llama:
    if model_id in _loaded_models:
        return _loaded_models[model_id]

    # Find the model file
    matches = list(MODELS_DIR.glob(f"{model_id}*.gguf"))
    if not matches:
        raise FileNotFoundError(f"Model '{model_id}' not found in {MODELS_DIR}")

    path = str(matches[0])
    print(f"Loading model: {path}")

    llm = Llama(
        model_path=path,
        n_ctx=2048,
        n_threads=os.cpu_count(),
        verbose=False
    )

    _loaded_models[model_id] = llm
    return llm

def generate_stream(model_id: str, messages: list[dict],
                    max_tokens: int, temperature: float, top_p: float):
    llm = get_model(model_id)

    # Format messages into a prompt using TinyLlama chat template
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"<|system|>\n{msg['content']}</s>\n"
        elif msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    prompt += "<|assistant|>\n"

    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        echo=False,
        stop=["</s>", "<|user|>", "<|system|>"]
    )

    for chunk in stream:
        token = chunk["choices"][0]["text"]
        if token:
            yield token