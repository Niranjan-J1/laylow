#include "gguf.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace laylow {

template<typename T>
static T read_val(std::ifstream& f) {
    T val;
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    if (!f) throw std::runtime_error("GGUF: unexpected end of file");
    return val;
}

static std::string read_string(std::ifstream& f) {
    uint64_t len = read_val<uint64_t>(f);
    if (len > 1024 * 1024)
        throw std::runtime_error("GGUF: string too long, likely corrupt");
    std::string s(len, '\0');
    f.read(s.data(), len);
    if (!f) throw std::runtime_error("GGUF: failed to read string");
    return s;
}

enum class GGUFType : uint32_t {
    UINT8=0, INT8=1, UINT16=2, INT16=3,
    UINT32=4, INT32=5, FLOAT32=6, BOOL=7,
    STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12,
};

enum class GGMLType : uint32_t {
    F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9,
    Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15,
};

static GGUFValue read_value(std::ifstream& f, GGUFType type) {
    switch (type) {
        case GGUFType::UINT8:   return read_val<uint8_t>(f);
        case GGUFType::INT8:    return read_val<int8_t>(f);
        case GGUFType::UINT16:  return read_val<uint16_t>(f);
        case GGUFType::INT16:   return read_val<int16_t>(f);
        case GGUFType::UINT32:  return read_val<uint32_t>(f);
        case GGUFType::INT32:   return read_val<int32_t>(f);
        case GGUFType::UINT64:  return read_val<uint64_t>(f);
        case GGUFType::INT64:   return read_val<int64_t>(f);
        case GGUFType::FLOAT32: return read_val<float>(f);
        case GGUFType::FLOAT64: return read_val<double>(f);
        case GGUFType::BOOL:    return static_cast<bool>(read_val<uint8_t>(f));
        case GGUFType::STRING:  return read_string(f);
        case GGUFType::ARRAY: {
            auto elem_type = static_cast<GGUFType>(read_val<uint32_t>(f));
            uint64_t count = read_val<uint64_t>(f);
            for (uint64_t i = 0; i < count; i++)
                read_value(f, elem_type);
            return std::string("[array]");
        }
        default:
            throw std::runtime_error("GGUF: unknown value type");
    }
}

static void extract_config(ModelConfig& cfg,
                            const std::string& key,
                            const GGUFValue& val)
{
    if (key == "general.architecture") {
        cfg.arch = std::get<std::string>(val);
        return;
    }

    auto as_u32 = [&]() -> uint32_t {
        if (std::holds_alternative<uint32_t>(val)) return std::get<uint32_t>(val);
        if (std::holds_alternative<uint64_t>(val)) return (uint32_t)std::get<uint64_t>(val);
        if (std::holds_alternative<int32_t>(val))  return (uint32_t)std::get<int32_t>(val);
        return 0;
    };

    if (key.find("vocab_size")              != std::string::npos) { cfg.n_vocab    = as_u32(); return; }
    if (key.find("context_length")          != std::string::npos) { cfg.n_ctx      = as_u32(); return; }
    if (key.find("embedding_length")        != std::string::npos) { cfg.n_embd     = as_u32(); return; }
    if (key.find("attention.head_count_kv") != std::string::npos) { cfg.n_heads_kv = as_u32(); return; }
    if (key.find("attention.head_count")    != std::string::npos &&
        key.find("kv") == std::string::npos)                      { cfg.n_heads    = as_u32(); return; }
    if (key.find("block_count")             != std::string::npos) { cfg.n_layers   = as_u32(); return; }

    if (key.find("attention.layer_norm_rms_epsilon") != std::string::npos)
        if (std::holds_alternative<float>(val))
            cfg.norm_eps = std::get<float>(val);
}

GGUFFile gguf_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("GGUF: cannot open file: " + path);

    uint32_t magic = read_val<uint32_t>(f);
    if (magic != 0x46554747)
        throw std::runtime_error("GGUF: invalid magic - not a GGUF file");

    uint32_t version = read_val<uint32_t>(f);
    if (version < 2 || version > 3)
        throw std::runtime_error("GGUF: unsupported version " + std::to_string(version));

    uint64_t n_tensors  = read_val<uint64_t>(f);
    uint64_t n_metadata = read_val<uint64_t>(f);

    std::cout << "GGUF v" << version << ": "
              << n_tensors << " tensors, "
              << n_metadata << " metadata entries" << std::endl;

    GGUFFile result;

    for (uint64_t i = 0; i < n_metadata; i++) {
        std::string key = read_string(f);
        auto type       = static_cast<GGUFType>(read_val<uint32_t>(f));
        GGUFValue val   = read_value(f, type);
        extract_config(result.config, key, val);
        result.metadata[key] = val;
    }

    struct TensorDesc {
        std::string          name;
        std::vector<int64_t> shape;
        DType                dtype;
        uint32_t             ggml_type_raw;
        uint64_t             offset;
        uint64_t             nbytes;
    };

    std::vector<TensorDesc> descs;
    descs.reserve(n_tensors);

    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorDesc d;
        d.name          = read_string(f);
        uint32_t n_dims = read_val<uint32_t>(f);

        for (uint32_t dim = 0; dim < n_dims; dim++)
            d.shape.push_back((int64_t)read_val<uint64_t>(f));

        std::reverse(d.shape.begin(), d.shape.end());

        d.ggml_type_raw = read_val<uint32_t>(f);
        auto ggml_type  = static_cast<GGMLType>(d.ggml_type_raw);
        switch (ggml_type) {
            case GGMLType::F32:  d.dtype = DType::F32;  break;
            case GGMLType::F16:  d.dtype = DType::F16;  break;
            case GGMLType::Q4_0: d.dtype = DType::Q4_0; break;
            case GGMLType::Q8_0: d.dtype = DType::Q8_0; break;
            case GGMLType::Q4_1:
            case GGMLType::Q5_0:
            case GGMLType::Q5_1:
            case GGMLType::Q2_K:
            case GGMLType::Q3_K:
            case GGMLType::Q4_K:
            case GGMLType::Q5_K:
            case GGMLType::Q6_K:
            case GGMLType::Q8_K:
            case GGMLType::Q8_1:
                d.dtype = DType::Q4_0;
                break;
            default:
                std::cerr << "GGUF: skipping unknown dtype "
                          << (uint32_t)ggml_type
                          << " for: " << d.name << std::endl;
                d.dtype = DType::F32;
                break;
        }
        d.offset = read_val<uint64_t>(f);

        // Calculate actual byte size on disk for this tensor
        int64_t n = 1;
        for (auto s : d.shape) n *= s;
        switch (d.ggml_type_raw) {
            case 0:  d.nbytes = n * 4; break;           // F32
            case 1:  d.nbytes = n * 2; break;           // F16
            case 2:  d.nbytes = (n / 32) * 18; break;   // Q4_0
            case 8:  d.nbytes = (n / 32) * 34; break;   // Q8_0
            case 10: d.nbytes = (n / 256) * 84; break;  // Q2_K
            case 11: d.nbytes = (n / 256) * 110; break; // Q3_K
            case 12: d.nbytes = (n / 256) * 144; break; // Q4_K
            case 13: d.nbytes = (n / 256) * 176; break; // Q5_K
            case 14: d.nbytes = (n / 256) * 210; break; // Q6_K
            default: d.nbytes = n * 4; break;
        }
        descs.push_back(std::move(d));
    }

    uint64_t pos         = (uint64_t)f.tellg();
    uint64_t aligned_pos = (pos + 31) & ~31ULL;
    f.seekg(aligned_pos);

    uint64_t data_start = (uint64_t)f.tellg();

    for (auto& d : descs) {
        Tensor t    = Tensor::empty(d.name, d.dtype, d.shape);
        t.nbytes    = d.nbytes;
        t.data      = _aligned_malloc(d.nbytes, 32);
        if (!t.data) throw std::runtime_error("GGUF: alloc failed: " + d.name);
        f.seekg(data_start + d.offset);
        f.read(static_cast<char*>(t.data), d.nbytes);
        if (!f) throw std::runtime_error("GGUF: failed to read tensor: " + d.name);
        result.tensors[d.name] = std::move(t);
    }

    std::cout << "Loaded " << result.tensors.size() << " tensors" << std::endl;
    std::cout << "Arch: "     << result.config.arch
              << "  layers: " << result.config.n_layers
              << "  embd: "   << result.config.n_embd
              << "  heads: "  << result.config.n_heads
              << "  vocab: "  << result.config.n_vocab << std::endl;

    return result;
}

} // namespace laylow
