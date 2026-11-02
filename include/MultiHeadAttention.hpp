#pragma once

#include "Module.hpp"
#include "Autograd.hpp"
#include "Ops.hpp"
#include "Linear.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>

class MultiHeadAttention : public Module {
public:
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    MultiHeadAttention(size_t embed_dim, size_t num_heads)
        : embed_dim(embed_dim),
          num_heads(num_heads),
          head_dim(embed_dim / num_heads),
          q_proj(embed_dim, embed_dim),
          k_proj(embed_dim, embed_dim),
          v_proj(embed_dim, embed_dim),
          out_proj(embed_dim, embed_dim) {
        if (embed_dim % num_heads != 0) {
            throw std::invalid_argument("embed_dim trebuie sa fie divizibil cu num_heads!");
        }
    }

    ValuePtr forward(const ValuePtr& x) {
        auto Q = q_proj.forward(x);
        auto K = k_proj.forward(x);
        auto V = v_proj.forward(x);

                auto K_T = EngineOps::transpose(K);

                auto scores = EngineOps::matmul(Q, K_T);

                float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
        auto scaled_scores = EngineOps::scale(scores, scale_factor);

                auto attn_weights = EngineOps::softmax(scaled_scores);

                auto context = EngineOps::matmul(attn_weights, V);

        return out_proj.forward(context);
    }

    std::vector<ValuePtr> parameters() const override {
        std::vector<ValuePtr> params;
        for (auto p : q_proj.parameters()) params.push_back(p);
        for (auto p : k_proj.parameters()) params.push_back(p);
        for (auto p : v_proj.parameters()) params.push_back(p);
        for (auto p : out_proj.parameters()) params.push_back(p);
        return params;
    }
};