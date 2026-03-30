#include "transformer.h"
#include "matmul.h"
#include "tensor.h"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace laylow {

static void rmsnorm(float* o, float* x, float* w, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = w[j] * (ss * x[j]);
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

// W (d,n) @ x (n,) -> xout (d,)
static void matmul(float* xout, const float* x, const float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
        xout[i] = val;
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

    auto embd_it = gguf.tensors.find("token_embd.weight");
    if (embd_it != gguf.tensors.end())
        cfg.n_vocab = (int)embd_it->second.dim(0);

    auto ffn_it = gguf.tensors.find("blk.0.ffn_gate.weight");
    if (ffn_it != gguf.tensors.end())
        cfg.n_ffn = (int)ffn_it->second.dim(0);
    else
        cfg.n_ffn = (cfg.n_embd * 8 / 3 + 127) / 128 * 128;

    std::cout << "Transformer config:" << std::endl;
    std::cout << "  vocab="    << cfg.n_vocab
              << " embd="     << cfg.n_embd
              << " heads="    << cfg.n_heads
              << " kv_heads=" << cfg.n_heads_kv
              << " layers="   << cfg.n_layers
              << " head_dim=" << cfg.head_dim
              << " n_ffn="    << cfg.n_ffn    << std::endl;

    auto get_f32 = [&](const std::string& name) -> float* {
        auto it = gguf.tensors.find(name);
        if (it == gguf.tensors.end()) {
            std::cerr << "Warning: tensor not found: " << name << std::endl;
            return nullptr;
        }
        const Tensor& t = it->second;
        if (t.dtype == DType::F32)
            return static_cast<float*>(t.data);
        Tensor f32;
        switch (t.dtype) {
            case DType::Q4_0: f32 = dequantize_q4(t);  break;
            case DType::Q8_0: f32 = dequantize_q8(t);  break;
            case DType::Q6_K: f32 = dequantize_q6k(t); break;
            default:
                std::cerr << "Warning: unsupported dtype for: " << name << std::endl;
                return nullptr;
        }
        dequantized_.push_back(std::move(f32));
        return static_cast<float*>(dequantized_.back().data);
    };

    weights.token_embd  = get_f32("token_embd.weight");
    weights.output_norm = get_f32("output_norm.weight");
    weights.output      = get_f32("output.weight");
    if (!weights.output) weights.output = weights.token_embd;

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

    std::cout << "Weights loaded for " << cfg.n_layers << " layers" << std::endl;
}

std::vector<float> Transformer::forward(const std::vector<int>& tokens) {
    const int seq_len   = (int)tokens.size();
    const int dim       = cfg.n_embd;
    const int n_heads   = cfg.n_heads;
    const int n_kv_heads= cfg.n_heads_kv;
    const int head_size = cfg.head_dim;
    const int kv_dim    = (dim * n_kv_heads) / n_heads;
    const int kv_mul    = n_heads / n_kv_heads;
    const int hidden_dim= cfg.n_ffn;

    // Allocate all buffers exactly like Karpathy
    std::vector<float> x(dim);
    std::vector<float> xb(dim);
    std::vector<float> xb2(dim);
    std::vector<float> hb(hidden_dim);
    std::vector<float> hb2(hidden_dim);
    std::vector<float> q(dim);
    std::vector<float> att(n_heads * seq_len);
    std::vector<float> logits(cfg.n_vocab, 0.0f);

    // KV cache: [n_layers * seq_len * kv_dim]
    std::vector<float> key_cache(cfg.n_layers * cfg.n_ctx * kv_dim, 0.0f);
    std::vector<float> val_cache(cfg.n_layers * cfg.n_ctx * kv_dim, 0.0f);

    // Process each token position — exactly like Karpathy's loop
    for (int pos = 0; pos < seq_len; pos++) {
        int token = tokens[pos];

        // Copy token embedding into x
        if (weights.token_embd) {
            float* content_row = weights.token_embd + token * dim;
            memcpy(x.data(), content_row, dim * sizeof(float));
        }

        // Forward all layers
        for (int l = 0; l < cfg.n_layers; l++) {
            auto& lw = weights.layers[l];
            if (!lw.attn_norm || !lw.attn_q || !lw.attn_k ||
                !lw.attn_v   || !lw.attn_out) continue;

            // Attention rmsnorm
            rmsnorm(xb.data(), x.data(), lw.attn_norm, dim);

            // KV cache layer offset
            int loff = l * cfg.n_ctx * kv_dim;

            // Pointers directly into KV cache for current position
            float* k = key_cache.data() + loff + pos * kv_dim;
            float* v = val_cache.data() + loff + pos * kv_dim;

            // QKV matmuls
            matmul(q.data(), xb.data(), lw.attn_q, dim, dim);
            matmul(k,        xb.data(), lw.attn_k, dim, kv_dim);
            matmul(v,        xb.data(), lw.attn_v, dim, kv_dim);

            // RoPE — exactly like Karpathy: iterate over dim, use i % head_size
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq   = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val    = pos * freq;
                float fcr    = cosf(val);
                float fci    = sinf(val);
                int rotn     = i < kv_dim ? 2 : 1;
                for (int rv = 0; rv < rotn; rv++) {
                    float* vec = rv == 0 ? q.data() : k;
                    float v0   = vec[i];
                    float v1   = vec[i + 1];
                    vec[i]     = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Multihead attention
            for (int h = 0; h < n_heads; h++) {
                float* qh  = q.data() + h * head_size;
                float* ath = att.data() + h * seq_len;

                // Attention scores
                for (int t = 0; t <= pos; t++) {
                    float* kh  = key_cache.data() + loff + t * kv_dim
                                 + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++)
                        score += qh[i] * kh[i];
                    score /= sqrtf((float)head_size);
                    ath[t] = score;
                }

                softmax(ath, pos + 1);

                // Weighted sum into xb
                float* xbh = xb.data() + h * head_size;
                memset(xbh, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* vh = val_cache.data() + loff + t * kv_dim
                                + (h / kv_mul) * head_size;
                    float a = ath[t];
                    for (int i = 0; i < head_size; i++)
                        xbh[i] += a * vh[i];
                }
            }

            // Output projection
            matmul(xb2.data(), xb.data(), lw.attn_out, dim, dim);

            // Residual
            for (int i = 0; i < dim; i++) x[i] += xb2[i];

            // FFN
            if (!lw.ffn_norm || !lw.ffn_gate || !lw.ffn_up || !lw.ffn_down)
                continue;

            rmsnorm(xb.data(), x.data(), lw.ffn_norm, dim);

            matmul(hb.data(),  xb.data(), lw.ffn_gate, dim, hidden_dim);
            matmul(hb2.data(), xb.data(), lw.ffn_up,   dim, hidden_dim);

            // SwiGLU: silu(gate) * up
            for (int i = 0; i < hidden_dim; i++) {
                float val = hb[i];
                val *= (1.0f / (1.0f + expf(-val))); // silu
                val *= hb2[i];
                hb[i] = val;
            }

            matmul(xb.data(), hb.data(), lw.ffn_down, hidden_dim, dim);

            // Residual
            for (int i = 0; i < dim; i++) x[i] += xb[i];
        }
    }

    // Final rmsnorm
    if (weights.output_norm)
        rmsnorm(x.data(), x.data(), weights.output_norm, dim);

    // Classifier
    matmul(logits.data(), x.data(), weights.output, dim, cfg.n_vocab);

    return logits;
}

int Transformer::sample_greedy(const std::vector<float>& logits) {
    return (int)(std::max_element(logits.begin(), logits.end())
                 - logits.begin());
}

int Transformer::sample_temperature(const std::vector<float>& logits, float temp) {
    int n = (int)logits.size();
    std::vector<float> probs(n);
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - max_val) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
    float r = (float)rand() / (RAND_MAX + 1.0f);
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

int Transformer::sample_topp(const std::vector<float>& logits,
                              float temp, float top_p)
{
    int n = (int)logits.size();
    std::vector<std::pair<float,int>> probs(n);
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = { expf((logits[i] - max_val) / temp), i };
        sum += probs[i].first;
    }
    for (int i = 0; i < n; i++) probs[i].first /= sum;
    std::sort(probs.begin(), probs.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });
    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i].first;
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }
    sum = 0.0f;
    for (int i = 0; i < cutoff; i++) sum += probs[i].first;
    for (int i = 0; i < cutoff; i++) probs[i].first /= sum;
    float r = (float)rand() / (RAND_MAX + 1.0f);
    cumsum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        cumsum += probs[i].first;
        if (r < cumsum) return probs[i].second;
    }
    return probs[0].second;
}

void Transformer::apply_rep_penalty(std::vector<float>& logits,
                                     const std::vector<int>& prev_tokens,
                                     float penalty)
{
    for (int id : prev_tokens) {
        if (id >= 0 && id < (int)logits.size()) {
            if (logits[id] > 0) logits[id] /= penalty;
            else logits[id] *= penalty;
        }
    }
}

} // namespace laylow