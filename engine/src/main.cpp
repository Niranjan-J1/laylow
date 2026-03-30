#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
#include "tensor.h"
#include "matmul.h"
#include "gguf.h"
#include "tokenizer.h"
#include "transformer.h"
#include <iomanip>

int main(int argc, char* argv[]) {
    std::cout << "laylow v0.1.0 - local LLM inference engine" << std::endl;
    std::cout << "AVX2 SIMD enabled" << std::endl;
    std::cout << std::endl;

    if (argc < 2) {
        std::cout << "Usage: laylow <model.gguf> [prompt] [max_tokens]" << std::endl;
        std::cout << "       laylow <model.gguf> --benchmark" << std::endl;
        return 0;
    }

    std::string model_path = argv[1];
    bool benchmark = argc > 2 && std::string(argv[2]) == "--benchmark";
    std::string prompt     = argc > 2 && !benchmark ? argv[2] : "Once upon a time";
    int max_new_tokens     = argc > 3 ? std::stoi(argv[3]) : 30;

    try {
        std::cout << "Loading model: " << model_path << std::endl;
        auto gguf = laylow::gguf_load(model_path);

        laylow::Tokenizer tok;
        tok.load_from_gguf(gguf.metadata);

        laylow::Transformer transformer;
        transformer.load(gguf);

        if (benchmark) {
            std::cout << std::endl;
            std::cout << "Running benchmark..." << std::endl;
            std::cout << std::endl;

            // Warmup run
            auto warmup_ids = tok.encode("Hello");
            transformer.forward(warmup_ids);

            // Benchmark: measure tokens per second
            std::vector<std::string> test_prompts = {
                "Once upon a time",
                "The meaning of life",
                "Python is a programming language"
            };

            int total_tokens = 0;
            double total_time = 0.0;

            for (const auto& p : test_prompts) {
                auto ids = tok.encode(p);
                int gen_tokens = 20;

                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < gen_tokens; i++) {
                    auto logits = transformer.forward(ids);
                    int next_id = transformer.sample_greedy(logits);
                    if (next_id == tok.eos_id) break;
                    ids.push_back(next_id);
                }

                auto end = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(end - start).count();

                double tps = gen_tokens / elapsed;
                total_tokens += gen_tokens;
                total_time   += elapsed;

                std::cout << "  Prompt: \"" << p << "\"" << std::endl;
                std::cout << "  Tokens: " << gen_tokens
                          << "  Time: " << std::fixed << std::setprecision(2)
                          << elapsed << "s"
                          << "  Speed: " << std::fixed << std::setprecision(1)
                          << tps << " tok/s" << std::endl;
                std::cout << std::endl;
            }

            double avg_tps = total_tokens / total_time;
            std::cout << "Average: " << std::fixed << std::setprecision(1)
                      << avg_tps << " tokens/second" << std::endl;
            std::cout << "Model:   " << model_path << std::endl;
            std::cout << "Threads: " << 1 << " (single threaded)" << std::endl;
            return 0;
        }

        // Normal generation
        std::string formatted =
            "<|system|>\nYou are a helpful assistant.</s>\n"
            "<|user|>\n" + prompt + "</s>\n<|assistant|>\n";

        auto ids = tok.encode(formatted);
        std::cout << "Prompt: \"" << prompt << "\""
                  << "  (" << ids.size() << " tokens)" << std::endl;
        std::cout << std::endl;

        std::cout << prompt;
        std::cout.flush();

        // Reset KV cache for new conversation
        transformer.reset_cache();
        
        srand(42);

        auto gen_start = std::chrono::high_resolution_clock::now();
        int gen_count  = 0;

        for (int i = 0; i < max_new_tokens; i++) {
            auto logits = transformer.forward(ids);
            transformer.apply_rep_penalty(logits, ids, 1.3f);
            int next_id = transformer.sample_topp(logits, 0.9f, 0.95f);

            if (next_id == tok.eos_id) {
                std::cout << std::endl << "[EOS]" << std::endl;
                break;
            }

            std::cout << tok.decode({next_id});
            std::cout.flush();
            ids.push_back(next_id);
            gen_count++;

            if ((int)ids.size() >= transformer.cfg.n_ctx) break;
        }

        auto gen_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(gen_end - gen_start).count();
        double tps = gen_count / elapsed;

        std::cout << std::endl << std::endl;
        std::cout << "Generated " << gen_count << " tokens"
                  << " at " << std::fixed << std::setprecision(1)
                  << tps << " tok/s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}