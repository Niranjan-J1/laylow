#pragma once
#include "tensor.h"
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <cstdint>

namespace laylow {

using GGUFValue = std::variant<
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    uint64_t, int64_t,
    float, double,
    bool,
    std::string
>;

struct ModelConfig {
    std::string arch;
    uint32_t    n_vocab    = 0;
    uint32_t    n_ctx      = 0;
    uint32_t    n_embd     = 0;
    uint32_t    n_heads    = 0;
    uint32_t    n_heads_kv = 0;
    uint32_t    n_layers   = 0;
    float       norm_eps   = 1e-5f;
};

struct GGUFFile {
    ModelConfig config;
    std::unordered_map<std::string, GGUFValue> metadata;
    std::unordered_map<std::string, Tensor>    tensors;
};

GGUFFile gguf_load(const std::string& path);

} // namespace laylow
