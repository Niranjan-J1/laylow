#include "transformer.h"
#include "matmul.h"
#include "tensor.h"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <cstring>

namespace laylow {

static void rms_norm(float* out, const float* x, const float* w,
                     int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * ss * w[i];
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void softmax(float* x, int n) {
    float max_val = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void matvec(float* out, const float* mat, const float* vec,
                   int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const float* row = mat + i * cols;
        for (int j = 0; j < cols; j++)
            sum += row[j] * vec[j];
        out[i] = sum;
    }
}

static void rope(float* vec, int pos, int head_dim) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq  = 1.0f / powf(10000.0f, (float)i / head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

void Transformer::load(const GGUFFile& gguf) {
    const auto& c = gguf.config;

    cfg.n_embd     = c.n_embd     ? c.n_embd     : 2048;
    cfg.n_heads    = c.n_heads    ? c.n_heads    : 32;
    cfg.n_heads_kv = c.n_heads_kv ? c.n_heads_kv : 4;
    cfg.n_layers   = c.n_layers   ? c.n_layers   : 22;
    cfg.n_ctx      = c.n_ctx      ? c.n_ctx      : 2048;
    cfg.norm_eps   = c.norm_eps;
    cfg.head_dim   = cfg.n_embd / cfg.n_heads;

    // Get vocab size from the embedding tensor directly
    auto embd_it = gguf.tensors.find("token_embd.weight");
    if (embd_it != gguf.tensors.end())
        cfg.n_vocab = (int)embd_it->second.dim(0);

    std::cout << "Transformer config:" << std::endl;
    std::cout << "  vocab="    << cfg.n_vocab
              << " embd="     << cfg.n_embd
              << " heads="    << cfg.n_heads
              << " kv_heads=" << cfg.n_heads_kv
              << " layers="   << cfg.n_layers
              << " head_dim=" << cfg.head_dim << std::endl;

    // Dequantize a tensor to float32 on the fly
    auto get_f32 = [&](const std::string& name) -> float* {
        auto it = gguf.tensors.find(name);
        if (it == gguf.tensors.end()) {
            std::cerr << "Warning: tensor not found: " << name << std::endl;
            return nullptr;
        }
        const Tensor& t = it->second;
        if (t.dtype == DType::F32) {
            return static_cast<float*>(t.data);
        }
        // Dequantize Q4 or Q8 to float32
        Tensor f32 = dequantize_q4(t);
        // Store in dequantized buffer so it stays alive
        dequantized_.push_back(std::move(f32));
        return static_cast<float*>(dequantized_.back().data);
    };

    weights.token_embd  = get_f32("token_embd.weight");
    weights.output_norm = get_f32("output_norm.weight");
    weights.output      = get_f32("output.weight");

    weights.layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; i++) {
        std::string b = "blk." + std::to_string(i) + ".";
        auto& l = weights.layers[i];
        l.attn_norm = get_f32(b + "attn_norm.weight");
        l.attn_q    = get_f32(b + "attn_q.weight");
        l.attn_k    = get_f32(b + "attn_k.weight");
        l.attn_v    = get_f32(b + "attn_v.weight");
        l.attn_out  = get_f32(b + "attn_output.weight");
        l.ffn_norm  = get_f32(b + "ffn_norm.weight");
        l.ffn_gate  = get_f32(b + "ffn_gate.weight");
        l.ffn_up    = get_f32(b + "ffn_up.weight");
        l.ffn_down  = get_f32(b + "ffn_down.weight");
    }

    std::cout << "Weights dequantized for " << cfg.n_layers
              << " layers" << std::endl;
}

std::vector<float> Transformer::forward(const std::vector<int>& tokens) {
    const int seq_len = (int)tokens.size();
    const int n_embd  = cfg.n_embd;
    const int n_heads = cfg.n_heads;
    const int n_kv    = cfg.n_heads_kv;
    const int hd      = cfg.head_dim;
    const int n_ffn   = (int)((n_embd * 8 / 3 + 127) / 128 * 128);

    std::vector<float> x(n_embd);
    std::vector<float> xb(n_embd);
    std::vector<float> q(n_heads * hd);
    std::vector<float> k(n_kv * hd);
    std::vector<float> v(n_kv * hd);
    std::vector<float> attn(n_heads * seq_len);
    std::vector<float> attn_out(n_embd);
    std::vector<float> ffn_buf(n_ffn);
    std::vector<float> ffn_gate(n_ffn);

    std::vector<std::vector<float>> k_cache(cfg.n_layers,
        std::vector<float>(cfg.n_ctx * n_kv * hd, 0.0f));
    std::vector<std::vector<float>> v_cache(cfg.n_layers,
        std::vector<float>(cfg.n_ctx * n_kv * hd, 0.0f));

    int pos = seq_len - 1;

    // Embedding lookup
    int token_id = tokens[pos];
    if (weights.token_embd) {
        const float* emb = weights.token_embd + token_id * n_embd;
        std::copy(emb, emb + n_embd, x.begin());
    }

    // Transformer layers
    for (int layer = 0; layer < cfg.n_layers; layer++) {
        auto& l = weights.layers[layer];
        if (!l.attn_norm || !l.attn_q || !l.attn_k ||
            !l.attn_v    || !l.attn_out) continue;

        // Attention
        rms_norm(xb.data(), x.data(), l.attn_norm, n_embd, cfg.norm_eps);

        matvec(q.data(), l.attn_q, xb.data(), n_heads * hd, n_embd);
        matvec(k.data(), l.attn_k, xb.data(), n_kv    * hd, n_embd);
        matvec(v.data(), l.attn_v, xb.data(), n_kv    * hd, n_embd);

        for (int h = 0; h < n_heads; h++)
            rope(q.data() + h * hd, pos, hd);
        for (int h = 0; h < n_kv; h++)
            rope(k.data() + h * hd, pos, hd);

        float* kc = k_cache[layer].data() + pos * n_kv * hd;
        float* vc = v_cache[layer].data() + pos * n_kv * hd;
        std::copy(k.begin(), k.end(), kc);
        std::copy(v.begin(), v.end(), vc);

        float scale = 1.0f / sqrtf((float)hd);
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);

        for (int h = 0; h < n_heads; h++) {
            int kv_h = h % n_kv;
            float* scores = attn.data() + h * seq_len;

            for (int t = 0; t <= pos; t++) {
                const float* kt = k_cache[layer].data()
                                  + t * n_kv * hd + kv_h * hd;
                float dot = 0.0f;
                const float* qh = q.data() + h * hd;
                for (int d = 0; d < hd; d++)
                    dot += qh[d] * kt[d];
                scores[t] = dot * scale;
            }

            softmax(scores, pos + 1);

            float* out_h = attn_out.data() + h * hd;
            for (int t = 0; t <= pos; t++) {
                const float* vt = v_cache[layer].data()
                                  + t * n_kv * hd + kv_h * hd;
                for (int d = 0; d < hd; d++)
                    out_h[d] += scores[t] * vt[d];
            }
        }

        matvec(xb.data(), l.attn_out, attn_out.data(), n_embd, n_embd);
        for (int i = 0; i < n_embd; i++) x[i] += xb[i];

        // FFN
        if (!l.ffn_norm || !l.ffn_gate || !l.ffn_up || !l.ffn_down)
            continue;

        rms_norm(xb.data(), x.data(), l.ffn_norm, n_embd, cfg.norm_eps);

        matvec(ffn_gate.data(), l.ffn_gate, xb.data(), n_ffn, n_embd);
        matvec(ffn_buf.data(),  l.ffn_up,   xb.data(), n_ffn, n_embd);

        for (int i = 0; i < n_ffn; i++)
            ffn_buf[i] = silu(ffn_gate[i]) * ffn_buf[i];

        matvec(xb.data(), l.ffn_down, ffn_buf.data(), n_embd, n_ffn);
        for (int i = 0; i < n_embd; i++) x[i] += xb[i];
    }

    // Final norm
    if (weights.output_norm)
        rms_norm(x.data(), x.data(), weights.output_norm, n_embd, cfg.norm_eps);

    // Project to vocab
    std::vector<float> logits(cfg.n_vocab, 0.0f);
    if (weights.output)
        matvec(logits.data(), weights.output, x.data(), cfg.n_vocab, n_embd);

    return logits;
}

int Transformer::sample_greedy(const std::vector<float>& logits) {
    return (int)(std::max_element(logits.begin(), logits.end())
                 - logits.begin());
}

} // namespace laylow