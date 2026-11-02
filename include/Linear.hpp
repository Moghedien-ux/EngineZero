#pragma once

#include "Module.hpp"
#include "Autograd.hpp"
#include "Ops.hpp"
#include <random>
#include <cmath>

class Linear : public Module { public:
    ValuePtr W;
    ValuePtr b;

    Linear(size_t in_features, size_t out_features) {
        float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-limit, limit);

        Tensor w_tensor({in_features, out_features});
        for (size_t i = 0; i < w_tensor.size(); ++i) {
            w_tensor.at_flat(i) = dis(gen);
        }

        Tensor b_tensor({1, out_features}, 0.0f);

        W = ValueNode::make(w_tensor, {}, "W");
        b = ValueNode::make(b_tensor, {}, "b");
    }

    ValuePtr forward(const ValuePtr& x) {
                auto xw = EngineOps::matmul(x, W);
        return EngineOps::add(xw, b);
    }

    std::vector<ValuePtr> parameters() const override {         return {W, b};
    }
};