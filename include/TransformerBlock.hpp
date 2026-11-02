#pragma once

#include "Module.hpp"
#include "Autograd.hpp"
#include "Ops.hpp"
#include "Linear.hpp"
#include "LayerNorm.hpp"
#include "MultiHeadAttention.hpp"
#include <vector>

class TransformerBlock : public Module {
public:
    LayerNorm ln1;
    MultiHeadAttention mha;
    LayerNorm ln2;

        Linear ffn1;     Linear ffn2;
    TransformerBlock(size_t embed_dim, size_t num_heads)
        : ln1(embed_dim),
          mha(embed_dim, num_heads),
          ln2(embed_dim),
          ffn1(embed_dim, 4 * embed_dim),
          ffn2(4 * embed_dim, embed_dim) {}

    ValuePtr forward(const ValuePtr& x) {
                auto norm_x1 = ln1.forward(x);
        auto attn_out = mha.forward(norm_x1);
        auto x1 = EngineOps::add(x, attn_out);
                auto norm_x2 = ln2.forward(x1);
        auto hidden = ffn1.forward(norm_x2);
        auto activated = EngineOps::gelu(hidden);
        auto ffn_out = ffn2.forward(activated);

                return EngineOps::add(x1, ffn_out);
    }

    std::vector<ValuePtr> parameters() const override {
        std::vector<ValuePtr> params;
        for (auto p : ln1.parameters()) params.push_back(p);
        for (auto p : mha.parameters()) params.push_back(p);
        for (auto p : ln2.parameters()) params.push_back(p);
        for (auto p : ffn1.parameters()) params.push_back(p);
        for (auto p : ffn2.parameters()) params.push_back(p);
        return params;
    }
};