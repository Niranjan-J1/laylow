#pragma once
#include "gguf.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace laylow {

struct Tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    int bos_id = 1;
    int eos_id = 2;
    int unk_id = 0;

    // BPE merge rules
    std::vector<std::string> merges_;
    std::unordered_map<std::string, int> merge_rank_;

    void load_from_gguf(const std::unordered_map<std::string, GGUFValue>& metadata);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    int vocab_size() const { return (int)id_to_token.size(); }
};

} // namespace laylow