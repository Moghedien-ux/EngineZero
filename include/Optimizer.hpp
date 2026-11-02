#pragma once

#include "Autograd.hpp"
#include <vector>

class SGD {
private:
    std::vector<ValuePtr> m_params;
    float m_lr;

public:
    SGD(std::vector<ValuePtr> params, float lr = 0.01f)
        : m_params(std::move(params)), m_lr(lr) {}

    void step() {
        for (auto& param : m_params) {
            size_t total = param->data.size();
            float* data_ptr = param->data.data();
            const float* grad_ptr = param->grad.data();

#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total; ++i) {
                data_ptr[i] -= m_lr * grad_ptr[i];
            }
        }
    }

    void zero_grad() {
        for (auto& param : m_params) {
            param->grad = Tensor(param->data.shape(), 0.0f);
        }
    }
};