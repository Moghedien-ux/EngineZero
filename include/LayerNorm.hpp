#pragma once

#include "Module.hpp"
#include "Autograd.hpp"
#include "Ops.hpp"
#include <cmath>
#include <stdexcept>

class LayerNorm : public Module {
public:
    ValuePtr gamma;
    ValuePtr beta;
    float eps;

    LayerNorm(size_t features, float epsilon = 1e-5f) : eps(epsilon) {
        Tensor g_tensor({1, features}, 1.0f);
        Tensor b_tensor({1, features}, 0.0f);

        gamma = ValueNode::make(g_tensor, {}, "gamma");
        beta = ValueNode::make(b_tensor, {}, "beta");
    }

    ValuePtr forward(const ValuePtr& x) {
        if (x->data.ndim() != 2) {
            throw std::invalid_argument("LayerNorm necesita input 2D [Batch/Seq, Features].");
        }

        size_t rows = x->data.shape()[0];
        size_t cols = x->data.shape()[1];

        if (cols < 1) {
            throw std::invalid_argument("LayerNorm: Numarul de feature-uri trebuie sa fie minim 1.");
        }

        Tensor norm_data(x->data.shape());
        const float* x_ptr = x->data.data();
        float* n_ptr = norm_data.data();

        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r) {
            size_t offset = r * cols;

                        double sum = 0.0;
            for (size_t c = 0; c < cols; ++c) {
                sum += static_cast<double>(x_ptr[offset + c]);
            }
            float mean = static_cast<float>(sum / static_cast<double>(cols));

                        double var_sum = 0.0;
            for (size_t c = 0; c < cols; ++c) {
                double diff = static_cast<double>(x_ptr[offset + c]) - static_cast<double>(mean);
                var_sum += diff * diff;
            }
            float var = static_cast<float>(var_sum / static_cast<double>(cols));

                        float inv_std = 1.0f / std::sqrt(var + eps);
            for (size_t c = 0; c < cols; ++c) {
                n_ptr[offset + c] = (x_ptr[offset + c] - mean) * inv_std;
            }
        }

        auto x_hat = ValueNode::make(norm_data, {x}, "layer_norm_hat");

        Tensor y_data(x->data.shape());
        const float* g_ptr = gamma->data.data();
        const float* b_ptr = beta->data.data();
        float* y_ptr = y_data.data();

        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r) {
            size_t offset = r * cols;
            for (size_t c = 0; c < cols; ++c) {
                y_ptr[offset + c] = n_ptr[offset + c] * g_ptr[c] + b_ptr[c];
            }
        }

        return ValueNode::make(y_data, {x_hat, gamma, beta}, "layer_norm");
    }

    std::vector<ValuePtr> parameters() const override {
        return {gamma, beta};
    }
};