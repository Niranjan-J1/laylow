# laylow

> Private, local AI that runs entirely on your CPU.  
> No GPU. No cloud. No data leaves your device — ever.

---

## Demo

*← drop your GIF here*

---

## Performance

Built with a custom C++ inference engine. All numbers measured on a consumer laptop CPU with no GPU.

| Metric | Value |
|---|---|
| Peak throughput | **99.6 tokens/sec** |
| Average throughput | **9.1 tokens/sec** |
| KV cache speedup | **895× over naive recomputation** |
| Long-prompt throughput | **2.1 tokens/sec** (40+ token context) |
| Model | TinyLlama 1.1B Q4_0 |
| Hardware | CPU only — 8 threads, no GPU |

### Throughput by prompt length

| Prompt length | Tokens/sec |
|---|---|
| Short (~10 tokens) | ~11.8 |
| Medium (~50 tokens) | 9.1 |
| Long (~500 tokens) | 2.1 |

### KV cache impact

| Method | Tokens/sec |
|---|---|
| Naive full-sequence recomputation | ~0.11 |
| Persistent KV cache | **99.6** |

> **895× speedup** by caching key/value projections across the generated sequence, eliminating redundant attention recomputation on every forward pass.

---

## What it is

Most "local AI" tools still phone home. laylow doesn't.

The entire inference stack runs on your CPU — a custom C++ transformer engine that reads quantized model files directly, with no Python ML libraries in the hot path. The React UI talks to a local FastAPI server, which talks to the engine. Nothing else.

Built for people who need genuinely private AI: healthcare workers, lawyers, developers, or anyone who'd rather not send their questions to a server somewhere.

---

## Quick start
```bash
git clone https://github.com/Niranjan-J1/laylow.git
cd laylow
pip install -r requirements.txt
python laylow.py
```

laylow downloads a model on first run and opens a chat interface at `http://localhost:3000`.

---

## Features

- **Private by default** — zero network requests after model download
- **Document Q&A (RAG)** — upload PDFs and ask questions about them
- **Memory system** — remembers facts across conversations
- **Conversation history** — sidebar with past chats
- **Model settings** — adjustable temperature and token limits
- **Multi-model support** — any GGUF model from HuggingFace

---

## Architecture
```
Browser  (React + Vite)
    │
    ▼
FastAPI REST server          /chat (SSE)  /models  /documents  /memory
    │
    ▼
Custom C++ inference engine  ◄─── laylow's core
    │
    ▼
Quantized GGUF model         (CPU only, no GPU required)
```

---

## The C++ engine

The `engine/` directory is a complete LLM inference engine written from scratch in C++17. No llama.cpp. No ONNX. No PyTorch.

| Component | Details |
|---|---|
| **GGUF parser** | Reads quantized model binaries; supports Q4_0 and Q6_K formats |
| **AVX2 SIMD matmul** | 256-bit registers, 8 floats/instruction — the core throughput driver |
| **Persistent KV cache** | Eliminates full-sequence recomputation; 895× measured speedup |
| **Multithreaded matmul** | Row-parallel computation across all CPU cores |
| **Transformer forward pass** | GQA attention, RoPE positional embeddings, RMSNorm, SwiGLU FFN |
| **BPE tokenizer** | 61,249 merge rules loaded directly from model metadata |
| **Top-p sampling** | With repetition penalty for coherent generation |

---

## Models tested

| Model | Parameters | Size | Speed |
|---|---|---|---|
| TinyLlama 1.1B Q4_0 | 1.1B | 637 MB | 9.1 tok/s |
| Phi-3 Mini Q4 | 3.8B | 2.2 GB | ~2 tok/s |

Any GGUF model from HuggingFace works. Place it in the `models/` folder.

---

## API

Server runs at `http://localhost:8080`.
```
POST   /chat           — SSE streaming chat
GET    /models         — list available models
POST   /documents      — upload PDF or TXT for RAG
GET    /memory         — view remembered facts
DELETE /memory         — clear memory
GET    /conversations  — conversation history
GET    /health         — server status
```

---

## Tech stack

| Layer | Stack |
|---|---|
| Inference engine | C++17, AVX2 SIMD intrinsics |
| API server | Python, FastAPI, uvicorn (SSE streaming) |
| Chat UI | React, Vite |
| Document retrieval | sentence-transformers, vector similarity search |
| Fallback inference | llama-cpp-python (unsupported model formats) |

---

## Requirements

- Python 3.10+
- Node.js 18+
- 8 GB RAM (16 GB recommended for Phi-3)
- Windows, macOS, or Linux

---

## License

MIT
