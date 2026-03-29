#include "tensor.h"
#include <stdexcept>
#include <numeric>
#include <cstring>
#include <cmath>

namespace laylow {

int64_t Tensor::numel() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1LL,
                           std::multiplies<int64_t>());
}

int64_t Tensor::dim(int i) const {
    if (i < 0 || i >= (int)shape.size())
        throw std::out_of_range("Tensor::dim index out of range");
    return shape[i];
}

bool Tensor::is_quantized() const {
    return dtype == DType::Q4_0 ||
           dtype == DType::Q8_0 ||
           dtype == DType::Q6_K;
}

static size_t dtype_nbytes(DType dtype, int64_t numel) {
    switch (dtype) {
        case DType::F32:  return numel * 4;
        case DType::F16:  return numel * 2;
        case DType::Q8_0: return (numel / 32)  * 34;
        case DType::Q4_0: return (numel / 32)  * 18;
        case DType::Q6_K: return (numel / 256) * 210;
        default: throw std::runtime_error("Unknown dtype");
    }
}

Tensor Tensor::empty(const std::string& name, DType dtype,
                     std::vector<int64_t> shape)
{
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
// float16 -> float32
// IEEE 754 half-precision conversion, handling denormals, inf, NaN.
// ---------------------------------------------------------------------
static float f16_to_f32(uint16_t h) {
    uint32_t sign     = (uint32_t)(h >> 15);
    uint32_t exponent = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mantissa = (uint32_t)(h & 0x3FF);
    uint32_t result;

    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign << 31;
        } else {
            // Denormal: normalize it
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            result = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf or NaN
        result = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
        result = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    }

    float f;
    memcpy(&f, &result, 4);
    return f;
}

// ---------------------------------------------------------------------
// Q4_0 dequantization
// Block layout (32 elements, 18 bytes):
//   [0..1]  f16 scale  (d)
//   [2..17] 16 bytes holding 32 x 4-bit values
//           byte[i]: low nibble = element i, high nibble = element i+16
// value = (nibble - 8) * d
// ---------------------------------------------------------------------
Tensor dequantize_q4(const Tensor& src) {
    int64_t n  = src.numel();
    auto out   = Tensor::empty(src.name + "_f32", DType::F32, src.shape);
    float* dst = static_cast<float*>(out.data);
    const uint8_t* raw = static_cast<const uint8_t*>(src.data);

    int64_t n_blocks = n / 32;

    for (int64_t b = 0; b < n_blocks; b++) {
        uint16_t scale_u16;
        memcpy(&scale_u16, raw, 2);
        raw += 2;
        float scale = f16_to_f32(scale_u16);

        float* blk = dst + b * 32;
        for (int i = 0; i < 16; i++) {
            uint8_t byte  = raw[i];
            blk[i]      = ((int)(byte & 0x0F) - 8) * scale;
            blk[i + 16] = ((int)(byte >> 4)   - 8) * scale;
        }
        raw += 16;
    }

    return out;
}

// ---------------------------------------------------------------------
// Q8_0 dequantization
// Block layout (32 elements, 34 bytes):
//   [0..1]  f16 scale  (d)
//   [2..33] 32 x int8 values
// value = int8_val * d
// ---------------------------------------------------------------------
Tensor dequantize_q8(const Tensor& src) {
    int64_t n  = src.numel();
    auto out   = Tensor::empty(src.name + "_f32", DType::F32, src.shape);
    float* dst = static_cast<float*>(out.data);
    const uint8_t* raw = static_cast<const uint8_t*>(src.data);

    int64_t n_blocks = n / 32;

    for (int64_t b = 0; b < n_blocks; b++) {
        uint16_t scale_u16;
        memcpy(&scale_u16, raw, 2);
        raw += 2;
        float scale = f16_to_f32(scale_u16);

        float* blk = dst + b * 32;
        for (int i = 0; i < 32; i++) {
            int8_t v;
            memcpy(&v, raw++, 1);
            blk[i] = v * scale;
        }
    }

    return out;
}

// ---------------------------------------------------------------------
// Q6_K dequantization  (matches llama.cpp dequantize_row_q6_K exactly)
//
// Super-block layout: 256 elements, 210 bytes
//   ql[128]  — low 4 bits of each 6-bit value, two values packed per byte
//              ql[i] low  nibble → element i      (i = 0..127)
//              ql[i] high nibble → element i+128  (i = 0..127)
//   qh[64]   — high 2 bits of each 6-bit value, four values packed per byte
//              qh[i] bits[1:0] → element i*4+0
//              qh[i] bits[3:2] → element i*4+1
//              qh[i] bits[5:4] → element i*4+2
//              qh[i] bits[7:6] → element i*4+3
//   sc[16]   — int8 sub-block scales (one per 16 elements, 16 sub-blocks total)
//   d[2]     — f16 super-scale
//
// Reconstruction:
//   6bit_val = (low4 | (high2 << 4)) - 32        range [-32, 31]
//   output   = 6bit_val * sc[sub_block] * d
//
// The super-block is split into 8 half-blocks of 32 elements each.
// Each half-block has its own sc[] entry.
// Half-blocks 0..3 read ql low  nibbles (elements   0..127).
// Half-blocks 4..7 read ql high nibbles (elements 128..255).
// qh is indexed as qh[element / 4], shifted by (element % 4) * 2.
// ---------------------------------------------------------------------
Tensor dequantize_q6k(const Tensor& src) {
    int64_t n  = src.numel();
    auto out   = Tensor::empty(src.name + "_f32", DType::F32, src.shape);
    float* dst = static_cast<float*>(out.data);
    const uint8_t* raw = static_cast<const uint8_t*>(src.data);

    int64_t n_blocks = n / 256;

    for (int64_t i = 0; i < n_blocks; i++) {
        uint16_t d_raw;
        memcpy(&d_raw, raw + 208, 2);
        const float d          = f16_to_f32(d_raw);
        const uint8_t* ql      = raw;
        const uint8_t* qh      = raw + 128;
        const int8_t*  sc      = reinterpret_cast<const int8_t*>(raw + 192);
        float* y               = dst + i * 256;

        // Process 256 elements in 2 chunks of 128
        for (int n128 = 0; n128 < 2; n128++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;

                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }

        raw += 210;
    }

    return out;
}
} // namespace laylow