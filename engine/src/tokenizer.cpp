#include "tokenizer.h"
#include "gguf.h"
#include <stdexcept>
#include <iostream>

namespace laylow {

void Tokenizer::load_from_gguf(
    const std::unordered_map<std::string, GGUFValue>& metadata)
{
    // GGUF stores the vocabulary as an array under this key.
    // Each element is a string token in order of its ID.
    // We stored arrays as "[array]" placeholder strings so we
    // need to find the tokens a different way - they are stored
    // as individual keys: tokenizer.ggml.tokens
    // TinyLlama/LLaMA stores them packed in the tensor data.
    // For now we build a simple character-level fallback vocab
    // and will load the real BPE vocab from tensor data next step.

    // Check if we have vocab size in metadata
    auto it = metadata.find("tokenizer.ggml.tokens");
    if (it != metadata.end()) {
        std::cout << "Found tokenizer.ggml.tokens in metadata" << std::endl;
    }

    // Build a basic printable ASCII vocab as fallback
    // This lets us test the pipeline before full BPE is wired up
    id_to_token.clear();
    token_to_id.clear();

    // Reserve slots 0-2 for special tokens
    id_to_token.push_back("<unk>");   // 0
    id_to_token.push_back("<s>");     // 1 - BOS
    id_to_token.push_back("</s>");    // 2 - EOS

    // Add printable ASCII characters as single-char tokens
    for (int c = 32; c < 127; c++) {
        std::string tok(1, (char)c);
        token_to_id[tok] = (int)id_to_token.size();
        id_to_token.push_back(tok);
    }

    // Build reverse lookup for special tokens
    token_to_id["<unk>"] = 0;
    token_to_id["<s>"]   = 1;
    token_to_id["</s>"]  = 2;

    std::cout << "Tokenizer ready: " << vocab_size()
              << " tokens (fallback ASCII vocab)" << std::endl;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;

    // Always start with BOS token
    ids.push_back(bos_id);

    // Simple character-level tokenization
    // Each character becomes one token
    for (char c : text) {
        std::string tok(1, c);
        auto it = token_to_id.find(tok);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unk_id);
        }
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;

    for (int id : ids) {
        // Skip special tokens in output
        if (id == bos_id || id == eos_id) continue;

        if (id >= 0 && id < (int)id_to_token.size()) {
            result += id_to_token[id];
        }
    }

    return result;
}

} // namespace laylow