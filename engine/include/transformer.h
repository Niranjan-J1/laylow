#pragma once
#include "tensor.h"
#include "gguf.h"
#include "tokenizer.h"
#include <vector>
#include <string>

namespace laylow {

struct TransformerWeights {
    // Token embedding table: [vocab_size x n_embd]
    float* token_embd   = nullptr;

    // Output norm and projection
    float* output_norm  = nullptr;
    float* output       = nullptr;

    // Per-layer weights (indexed by layer)
    struct Layer {
        float* attn_norm  = nullptr; // RMS norm before attention
        float* attn_q     = nullptr; // Query projection
        float* attn_k     = nullptr; // Key projection
        float* attn_v     = nullptr; // Value projection
        float* attn_out   = nullptr; // Output projection
        float* ffn_norm   = nullptr; // RMS norm before FFN
        float* ffn_gate   = nullptr; // FFN gate projection
        float* ffn_up     = nullptr; // FFN up projection
        float* ffn_down   = nullptr; // FFN down projection
    };

    std::vector<Layer> layers;
};

struct TransformerConfig {
    int n_vocab   = 0;
    int n_ctx     = 0;
    int n_embd    = 0;
    int n_heads   = 0;
    int n_heads_kv= 0;
    int n_layers  = 0;
    int head_dim  = 0;  // n_embd / n_heads
    float norm_eps= 1e-5f;
};

struct Transformer {
    TransformerConfig  cfg;
    TransformerWeights weights;

    // Load weights from a parsed GGUF file
    void load(const GGUFFile& gguf);

    // Run one forward pass, return logits over vocabulary
    // tokens: sequence of token IDs
    std::vector<float> forward(const std::vector<int>& tokens);

    // Sample the next token from logits (greedy - pick highest)
    int sample_greedy(const std::vector<float>& logits);

    std::vector<Tensor> dequantized_; // holds dequantized weight buffers
};

} // namespace laylow