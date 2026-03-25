#include <iostream>
#include <cmath>
#include "tensor.h"
#include "matmul.h"
#include "gguf.h"
#include "tokenizer.h"
#include "transformer.h"

int main(int argc, char* argv[]) {
    std::cout << "laylow v0.1.0 - local LLM inference engine" << std::endl;
    std::cout << "AVX2 SIMD enabled" << std::endl;
    std::cout << std::endl;

    if (argc > 1) {
        std::string path = argv[1];
        std::cout << "Loading model: " << path << std::endl;

        try {
            // Load model
            auto gguf = laylow::gguf_load(path);

            // Load tokenizer
            laylow::Tokenizer tok;
            tok.load_from_gguf(gguf.metadata);

            // Load transformer weights
            laylow::Transformer transformer;
            transformer.load(gguf);

            // Encode a prompt
            std::string prompt = "Hello";
            auto ids = tok.encode(prompt);

            std::cout << std::endl;
            std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
            std::cout << "Running forward pass..." << std::endl;

            // Run forward pass
            auto logits = transformer.forward(ids);

            // Pick next token
            int next_id = transformer.sample_greedy(logits);
            std::string next_tok = tok.decode({next_id});

            std::cout << "Next token ID: " << next_id << std::endl;
            std::cout << "Next token:    \"" << next_tok << "\"" << std::endl;
            std::cout << std::endl;
            std::cout << "Forward pass complete" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    }

    // Matmul test
    int M = 32, K = 64, N = 32;
    auto A      = laylow::Tensor::empty("A",        laylow::DType::F32, {M, K});
    auto B      = laylow::Tensor::empty("B",        laylow::DType::F32, {K, N});
    auto C_ref  = laylow::Tensor::empty("C_scalar", laylow::DType::F32, {M, N});
    auto C_fast = laylow::Tensor::empty("C_avx2",   laylow::DType::F32, {M, N});

    float* a = static_cast<float*>(A.data);
    float* b = static_cast<float*>(B.data);
    for (int i = 0; i < M * K; i++) a[i] = i * 0.01f;
    for (int i = 0; i < K * N; i++) b[i] = i * 0.01f;

    laylow::matmul_scalar(a, b, static_cast<float*>(C_ref.data),  M, K, N);
    laylow::matmul_avx2  (a, b, static_cast<float*>(C_fast.data), M, K, N);

    bool ok = true;
    float* r  = static_cast<float*>(C_ref.data);
    float* ff = static_cast<float*>(C_fast.data);
    for (int i = 0; i < M * N; i++) {
        if (std::fabs(r[i] - ff[i]) > 1e-1f) { ok = false; break; }
    }

    std::cout << "matmul [" << M << "x" << K << "] * ["
              << K << "x" << N << "] = [" << M << "x" << N << "]" << std::endl;
    std::cout << "scalar vs AVX2: " << (ok ? "MATCH" : "FAIL") << std::endl;

    A.free_data(); B.free_data(); C_ref.free_data(); C_fast.free_data();
    return ok ? 0 : 1;
}