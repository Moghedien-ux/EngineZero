#pragma once

#include "Module.hpp"
#include "Autograd.hpp"
#include "Ops.hpp"
#include <random>
#include <cmath>

class Conv2D : public Module {
public:
    ValuePtr kernel;     size_t in_channels;
    size_t out_channels;
    size_t kernel_size;

    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size = 3)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size) {

        size_t k_flat_dim = in_channels * kernel_size * kernel_size;
        float limit = std::sqrt(6.0f / static_cast<float>(k_flat_dim + out_channels));

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-limit, limit);

        Tensor k_tensor({out_channels, k_flat_dim});
        for (size_t i = 0; i < k_tensor.size(); ++i) {
            k_tensor.at_flat(i) = dis(gen);
        }

        kernel = ValueNode::make(k_tensor, {}, "kernel");
    }

    ValuePtr forward(const ValuePtr& x) {
        return EngineOps::conv2d(x, kernel, 1, 0);
    }

    std::vector<ValuePtr> parameters() const override {
        return {kernel};
    }
};