#include <iostream>
#include <cmath>
#include <string>
#include "tensor.h"
#include "matmul.h"
#include "gguf.h"
#include "tokenizer.h"
#include "transformer.h"

int main(int argc, char* argv[]) {
    std::cout << "laylow v0.1.0 - local LLM inference engine" << std::endl;
    std::cout << "AVX2 SIMD enabled" << std::endl;
    std::cout << std::endl;

    if (argc < 2) {
        std::cout << "Usage: laylow <model.gguf> [prompt] [max_tokens]" << std::endl;
        std::cout << "Example: laylow tinyllama.gguf \"Once upon a time\" 50" << std::endl;
        return 0;
    }

    std::string model_path = argv[1];
    std::string prompt     = argc > 2 ? argv[2] : "Once upon a time";
    int max_new_tokens     = argc > 3 ? std::stoi(argv[3]) : 30;

    try {
        // Load model
        std::cout << "Loading model: " << model_path << std::endl;
        auto gguf = laylow::gguf_load(model_path);

        // Load tokenizer
        laylow::Tokenizer tok;
        tok.load_from_gguf(gguf.metadata);

        // Load transformer
        laylow::Transformer transformer;
        transformer.load(gguf);

        // Encode prompt
        auto ids = tok.encode(prompt);
        std::cout << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << "Prompt tokens: " << ids.size() << std::endl;
        std::cout << std::endl;
        std::cout << prompt;
        std::cout.flush();

        // Generation loop
        for (int i = 0; i < max_new_tokens; i++) {
            // Forward pass on current token sequence
            auto logits = transformer.forward(ids);

            // Pick next token
            int next_id = transformer.sample_greedy(logits);

            // Stop if we hit end of sequence
            if (next_id == tok.eos_id) {
                std::cout << std::endl;
                std::cout << "[EOS]" << std::endl;
                break;
            }

            // Decode and print the new token immediately
            std::string next_tok = tok.decode({next_id});
            std::cout << next_tok;
            std::cout.flush();

            // Add to sequence for next iteration
            ids.push_back(next_id);
        }

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Generated " << max_new_tokens
                  << " tokens" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}