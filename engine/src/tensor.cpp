#include "tensor.h"
#include <stdexcept>
#include <numeric>

namespace laylow {

int64_t Tensor::numel() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

int64_t Tensor::dim(int i) const {
    if (i < 0 || i >= (int)shape.size())
        throw std::out_of_range("Tensor::dim index out of range");
    return shape[i];
}

bool Tensor::is_quantized() const {
    return dtype == DType::Q4_0 || dtype == DType::Q8_0;
}

static size_t dtype_nbytes(DType dtype, int64_t numel) {
    switch (dtype) {
        case DType::F32:  return numel * 4;
        case DType::F16:  return numel * 2;
        case DType::Q8_0: return numel;
        case DType::Q4_0: return (numel / 2) + (numel / 32) * 4;
        default: throw std::runtime_error("Unknown dtype");
    }
}

Tensor Tensor::empty(const std::string& name, DType dtype, std::vector<int64_t> shape) {
    Tensor t;
    t.name   = name;
    t.dtype  = dtype;
    t.shape  = shape;
    t.nbytes = dtype_nbytes(dtype, t.numel());
    t.data   = _aligned_malloc(t.nbytes, 32);
    if (!t.data)
        throw std::runtime_error("Failed to allocate tensor: " + name);
    return t;
}

void Tensor::free_data() {
    if (data) {
        _aligned_free(data);
        data = nullptr;
    }
}

// ---------------------------------------------------------------------
// Q4_0 dequantization
// Format: blocks of 32 values
// Each block has: 2 bytes (float16 scale) + 16 bytes (32 x 4-bit weights)
// To decode: value = (nibble - 8) * scale
// ---------------------------------------------------------------------
Tensor dequantize_q4(const Tensor& src) {
    int64_t n = src.numel();
    auto out  = Tensor::empty(src.name + "_f32", DType::F32, src.shape);
    float* dst = static_cast<float*>(out.data);
    const uint8_t* raw = static_cast<const uint8_t*>(src.data);

    const int block_size = 32;
    int n_blocks = (int)(n / block_size);

    for (int b = 0; b < n_blocks; b++) {
        // First 2 bytes of block are float16 scale
        // We read it as uint16 and convert manually
        uint16_t scale_u16;
        memcpy(&scale_u16, raw, 2);
        raw += 2;

        // Convert float16 to float32
        uint32_t exp  = (scale_u16 >> 10) & 0x1F;
        uint32_t mant = scale_u16 & 0x3FF;
        uint32_t sign = (scale_u16 >> 15) & 1;
        float scale;
        if (exp == 0) {
            scale = (mant / 1024.0f) * (1.0f / 16384.0f);
        } else {
            uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&scale, &f32, 4);
        }

        // Next 16 bytes hold 32 x 4-bit values (2 per byte)
        float* out_block = dst + b * block_size;
        for (int i = 0; i < 16; i++) {
            uint8_t byte = *raw++;
            // Low nibble is first value, high nibble is second
            int lo = (byte & 0x0F) - 8;
            int hi = (byte >> 4)   - 8;
            out_block[i]      = lo * scale;
            out_block[i + 16] = hi * scale;
        }
    }

    return out;
}

// ---------------------------------------------------------------------
// Q8_0 dequantization
// Format: blocks of 32 values
// Each block has: 2 bytes (float16 scale) + 32 bytes (32 x int8 weights)
// To decode: value = int8_value * scale
// ---------------------------------------------------------------------
Tensor dequantize_q8(const Tensor& src) {
    int64_t n = src.numel();
    auto out   = Tensor::empty(src.name + "_f32", DType::F32, src.shape);
    float* dst = static_cast<float*>(out.data);
    const uint8_t* raw = static_cast<const uint8_t*>(src.data);

    const int block_size = 32;
    int n_blocks = (int)(n / block_size);

    for (int b = 0; b < n_blocks; b++) {
        // Read float16 scale
        uint16_t scale_u16;
        memcpy(&scale_u16, raw, 2);
        raw += 2;

        uint32_t exp  = (scale_u16 >> 10) & 0x1F;
        uint32_t mant = scale_u16 & 0x3FF;
        uint32_t sign = (scale_u16 >> 15) & 1;
        float scale;
        if (exp == 0) {
            scale = (mant / 1024.0f) * (1.0f / 16384.0f);
        } else {
            uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&scale, &f32, 4);
        }

        // 32 x int8 values
        float* out_block = dst + b * block_size;
        for (int i = 0; i < block_size; i++) {
            int8_t v;
            memcpy(&v, raw++, 1);
            out_block[i] = v * scale;
        }
    }

    return out;
}

} // namespace laylow