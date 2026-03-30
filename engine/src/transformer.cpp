#include "transformer.h"
#include "matmul.h"
#include "tensor.h"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <thread>

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
    // Split rows across threads
    const int n_threads = 8; // tune to your CPU core count
    std::vector<std::thread> threads;
    int chunk = (d + n_threads - 1) / n_threads;

    for (int t = 0; t < n_threads; t++) {
        int start = t * chunk;
        int end   = std::min(start + chunk, d);
        if (start >= d) break;

        threads.emplace_back([=]() {
            for (int i = start; i < end; i++) {
                float val = 0.0f;
                const float* row = w + i * n;
                for (int j = 0; j < n; j++) val += row[j] * x[j];
                xout[i] = val;
            }
        });
    }

    for (auto& th : threads) th.join();
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

void Transformer::init_cache() {
    int kv_dim = (cfg.n_embd * cfg.n_heads_kv) / cfg.n_heads;
    key_cache_.assign(cfg.n_layers * cfg.n_ctx * kv_dim, 0.0f);
    val_cache_.assign(cfg.n_layers * cfg.n_ctx * kv_dim, 0.0f);
    cache_pos_ = 0;
}

void Transformer::reset_cache() {
    cache_pos_ = 0;
}

std::vector<float> Transformer::forward(const std::vector<int>& tokens) {
    const int dim        = cfg.n_embd;
    const int n_heads    = cfg.n_heads;
    const int n_kv_heads = cfg.n_heads_kv;
    const int head_size  = cfg.head_dim;
    const int kv_dim     = (dim * n_kv_heads) / n_heads;
    const int kv_mul     = n_heads / n_kv_heads;
    const int hidden_dim = cfg.n_ffn;

    // Initialize cache on first use
    if (key_cache_.empty()) init_cache();

    // Determine which tokens to process
    // If cache is empty, process all tokens (prompt)
    // If cache has content, only process new tokens
    int start_pos = cache_pos_;
    int seq_len   = (int)tokens.size();

    std::vector<float> x(dim);
    std::vector<float> xb(dim);
    std::vector<float> xb2(dim);
    std::vector<float> hb(hidden_dim);
    std::vector<float> hb2(hidden_dim);
    std::vector<float> q(dim);
    std::vector<float> att(n_heads * cfg.n_ctx);
    std::vector<float> logits(cfg.n_vocab, 0.0f);

    // Process only new tokens
    for (int pos = start_pos; pos < seq_len; pos++) {
        int token = tokens[pos];

        if (weights.token_embd) {
            float* row = weights.token_embd + token * dim;
            memcpy(x.data(), row, dim * sizeof(float));
        }

        for (int l = 0; l < cfg.n_layers; l++) {
            auto& lw = weights.layers[l];
            if (!lw.attn_norm || !lw.attn_q || !lw.attn_k ||
                !lw.attn_v   || !lw.attn_out) continue;

            rmsnorm(xb.data(), x.data(), lw.attn_norm, dim);

            int loff = l * cfg.n_ctx * kv_dim;
            float* k = key_cache_.data() + loff + pos * kv_dim;
            float* v = val_cache_.data() + loff + pos * kv_dim;

            matmul(q.data(), xb.data(), lw.attn_q, dim, dim);
            matmul(k,        xb.data(), lw.attn_k, dim, kv_dim);
            matmul(v,        xb.data(), lw.attn_v, dim, kv_dim);

            // RoPE
            for (int i = 0; i < dim; i += 2) {
                int hd   = i % head_size;
                float freq = 1.0f / powf(10000.0f, hd / (float)head_size);
                float val  = pos * freq;
                float fcr  = cosf(val);
                float fci  = sinf(val);
                int rotn   = i < kv_dim ? 2 : 1;
                for (int rv = 0; rv < rotn; rv++) {
                    float* vec = rv == 0 ? q.data() : k;
                    float v0 = vec[i], v1 = vec[i+1];
                    vec[i]   = v0*fcr - v1*fci;
                    vec[i+1] = v0*fci + v1*fcr;
                }
            }

            // Attention over all cached positions
            for (int h = 0; h < n_heads; h++) {
                float* qh  = q.data() + h * head_size;
                float* ath = att.data() + h * cfg.n_ctx;

                for (int t = 0; t <= pos; t++) {
                    float* kh = key_cache_.data() + loff + t * kv_dim
                                + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++)
                        score += qh[i] * kh[i];
                    ath[t] = score / sqrtf((float)head_size);
                }

                softmax(ath, pos + 1);

                float* xbh = xb.data() + h * head_size;
                memset(xbh, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* vh = val_cache_.data() + loff + t * kv_dim
                                + (h / kv_mul) * head_size;
                    float a = ath[t];
                    for (int i = 0; i < head_size; i++)
                        xbh[i] += a * vh[i];
                }
            }

            matmul(xb2.data(), xb.data(), lw.attn_out, dim, dim);
            for (int i = 0; i < dim; i++) x[i] += xb2[i];

            if (!lw.ffn_norm || !lw.ffn_gate || !lw.ffn_up || !lw.ffn_down)
                continue;

            rmsnorm(xb.data(), x.data(), lw.ffn_norm, dim);
            matmul(hb.data(),  xb.data(), lw.ffn_gate, dim, hidden_dim);
            matmul(hb2.data(), xb.data(), lw.ffn_up,   dim, hidden_dim);

            for (int i = 0; i < hidden_dim; i++) {
                float val = hb[i];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= hb2[i];
                hb[i] = val;
            }

            matmul(xb.data(), hb.data(), lw.ffn_down, hidden_dim, dim);
            for (int i = 0; i < dim; i++) x[i] += xb[i];
        }

        cache_pos_ = pos + 1;
    }

    if (weights.output_norm)
        rmsnorm(x.data(), x.data(), weights.output_norm, dim);

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