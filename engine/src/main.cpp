#include <iostream>
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
        std::cout << "Loading model: " << model_path << std::endl;
        auto gguf = laylow::gguf_load(model_path);

        laylow::Tokenizer tok;
        tok.load_from_gguf(gguf.metadata);

        laylow::Transformer transformer;
        transformer.load(gguf);

        auto ids = tok.encode(prompt);
        
        std::cout << "Prompt: \"" << prompt << "\""
                  << "  (" << ids.size() << " tokens)" << std::endl;
        std::cout << std::endl;

        // Stream the prompt text first, then generated tokens
        std::cout << prompt;
        std::cout.flush();

        srand(42);

        for (int i = 0; i < max_new_tokens; i++) {
            auto logits = transformer.forward(ids);
            transformer.apply_rep_penalty(logits, ids, 1.3f);
            int next_id = transformer.sample_topp(logits, 1.1f, 0.95f);

            if (next_id == tok.eos_id) {
                std::cout << std::endl << "[EOS]" << std::endl;
                break;
            }

            std::cout << tok.decode({next_id});
            std::cout.flush();

            ids.push_back(next_id);

            if ((int)ids.size() >= transformer.cfg.n_ctx) break;
        }

        std::cout << std::endl << std::endl;
        std::cout << "Generated " << max_new_tokens << " tokens" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}