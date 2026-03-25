#include "tokenizer.h"
#include "gguf.h"
#include <stdexcept>
#include <iostream>
#include <sstream>

namespace laylow {

void Tokenizer::load_from_gguf(
    const std::unordered_map<std::string, GGUFValue>& metadata)
{
    id_to_token.clear();
    token_to_id.clear();

    // Look for the token array stored under this key
    auto it = metadata.find("tokenizer.ggml.tokens");
    if (it == metadata.end()) {
        std::cerr << "Warning: no tokenizer.ggml.tokens found, "
                     "using fallback ASCII vocab" << std::endl;
        // Fallback ASCII vocab
        id_to_token.push_back("<unk>");
        id_to_token.push_back("<s>");
        id_to_token.push_back("</s>");
        for (int c = 32; c < 127; c++)
            id_to_token.push_back(std::string(1, (char)c));
        for (int i = 0; i < (int)id_to_token.size(); i++)
            token_to_id[id_to_token[i]] = i;
        std::cout << "Tokenizer ready: " << vocab_size()
                  << " tokens (fallback)" << std::endl;
        return;
    }

    // The array was stored as a single string joined by \x01 separators
    const std::string& joined = std::get<std::string>(it->second);

    // Split on \x01 to recover individual tokens
    std::string tok;
    int id = 0;
    for (char c : joined) {
        if (c == '\x01') {
            id_to_token.push_back(tok);
            token_to_id[tok] = id++;
            tok.clear();
        } else {
            tok += c;
        }
    }
    // Last token
    if (!tok.empty()) {
        id_to_token.push_back(tok);
        token_to_id[tok] = id;
    }

    // Find special token IDs
    auto find_id = [&](const std::string& s) {
        auto it = token_to_id.find(s);
        return it != token_to_id.end() ? it->second : -1;
    };

    int bos = find_id("<s>");
    int eos = find_id("</s>");
    int unk = find_id("<unk>");
    if (bos >= 0) bos_id = bos;
    if (eos >= 0) eos_id = eos;
    if (unk >= 0) unk_id = unk;

    std::cout << "Tokenizer ready: " << vocab_size() << " tokens"
              << "  bos=" << bos_id
              << "  eos=" << eos_id << std::endl;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    ids.push_back(bos_id);

    // Greedy longest-match tokenization
    int i = 0;
    while (i < (int)text.size()) {
        // Try matching the longest token starting at position i
        int best_len = -1;
        int best_id  = unk_id;

        for (int len = (int)text.size() - i; len >= 1; len--) {
            std::string sub = text.substr(i, len);
            auto it = token_to_id.find(sub);
            if (it != token_to_id.end()) {
                best_len = len;
                best_id  = it->second;
                break;
            }
        }

        if (best_len < 1) {
            ids.push_back(unk_id);
            i++;
        } else {
            ids.push_back(best_id);
            i += best_len;
        }
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id == bos_id || id == eos_id) continue;
        if (id >= 0 && id < (int)id_to_token.size()) {
            std::string tok = id_to_token[id];
            // LLaMA uses \u2581 (▁) as a space prefix
            // Replace it with a real space
            size_t pos = 0;
            while ((pos = tok.find("\xe2\x96\x81", pos)) != std::string::npos) {
                tok.replace(pos, 3, " ");
                pos++;
            }
            result += tok;
        }
    }
    return result;
}

} // namespace laylow