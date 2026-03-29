#include "tokenizer.h"
#include "gguf.h"
#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <climits>

namespace laylow {

void Tokenizer::load_from_gguf(
    const std::unordered_map<std::string, GGUFValue>& metadata)
{
    id_to_token.clear();
    token_to_id.clear();
    merges_.clear();

    // Load vocabulary
    auto it = metadata.find("tokenizer.ggml.tokens");
    if (it == metadata.end()) {
        std::cerr << "Warning: no tokenizer.ggml.tokens found" << std::endl;
        return;
    }

    const std::string& joined = std::get<std::string>(it->second);
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
    if (!tok.empty()) {
        id_to_token.push_back(tok);
        token_to_id[tok] = id;
    }

    // Resolve special tokens
    auto find_id = [&](const std::string& s) {
        auto jt = token_to_id.find(s);
        return jt != token_to_id.end() ? jt->second : -1;
    };
    int bos = find_id("<s>");
    int eos = find_id("</s>");
    int unk = find_id("<unk>");
    if (bos >= 0) bos_id = bos;
    if (eos >= 0) eos_id = eos;
    if (unk >= 0) unk_id = unk;

    // Load BPE merge rules
    auto mit = metadata.find("tokenizer.ggml.merges");
    if (mit != metadata.end()) {
        const std::string& mjoined = std::get<std::string>(mit->second);
        std::string merge;
        for (char c : mjoined) {
            if (c == '\x01') {
                merges_.push_back(merge);
                merge.clear();
            } else {
                merge += c;
            }
        }
        if (!merge.empty()) merges_.push_back(merge);
    }

    // Build merge rank map: "a b" -> rank (lower = higher priority)
    for (int i = 0; i < (int)merges_.size(); i++)
        merge_rank_[merges_[i]] = i;

    std::cout << "Tokenizer ready: " << vocab_size() << " tokens"
              << "  bos=" << bos_id
              << "  eos=" << eos_id
              << "  merges=" << merges_.size() << std::endl;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    ids.push_back(bos_id);

    // Replace spaces with ▁ and prepend ▁ to start
    std::string normalized = "\xe2\x96\x81";
    for (char c : text) {
        if (c == ' ')
            normalized += "\xe2\x96\x81";
        else
            normalized += c;
    }

    // Split into individual UTF-8 characters as initial tokens
    // Each character starts as its own symbol
    std::vector<std::string> symbols;
    int i = 0;
    while (i < (int)normalized.size()) {
        unsigned char c = normalized[i];
        int char_len = 1;
        if      ((c & 0x80) == 0x00) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        symbols.push_back(normalized.substr(i, char_len));
        i += char_len;
    }

    // Apply BPE merges greedily by rank
    // Keep merging the highest-priority (lowest rank) pair until no merges apply
    while (symbols.size() > 1) {
        int best_rank = INT_MAX;
        int best_pos  = -1;

        for (int j = 0; j < (int)symbols.size() - 1; j++) {
            std::string pair = symbols[j] + " " + symbols[j + 1];
            auto rit = merge_rank_.find(pair);
            if (rit != merge_rank_.end() && rit->second < best_rank) {
                best_rank = rit->second;
                best_pos  = j;
            }
        }

        if (best_pos < 0) break; // no more merges

        // Apply the merge at best_pos
        symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    // Convert symbols to token IDs
    for (const std::string& sym : symbols) {
        auto jt = token_to_id.find(sym);
        if (jt != token_to_id.end()) {
            ids.push_back(jt->second);
        } else {
            // Byte fallback
            for (unsigned char byte : sym) {
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", byte);
                auto ht = token_to_id.find(std::string(hex));
                ids.push_back(ht != token_to_id.end() ? ht->second : unk_id);
            }
        }
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id == bos_id || id == eos_id) continue;
        if (id < 0 || id >= (int)id_to_token.size()) continue;

        std::string tok = id_to_token[id];

        // Byte-fallback tokens: <0xNN> -> raw byte
        if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' &&
            tok[2] == 'x' && tok[5] == '>') {
            char hex[3] = { tok[3], tok[4], 0 };
            result += (char)(unsigned char)strtol(hex, nullptr, 16);
            continue;
        }

        // Replace ▁ with space
        size_t pos = 0;
        while ((pos = tok.find("\xe2\x96\x81", pos)) != std::string::npos) {
            tok.replace(pos, 3, " ");
            pos++;
        }

        result += tok;
    }
    return result;
}

} // namespace laylow