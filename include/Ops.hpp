#pragma once

#include "Tensor.hpp"
#include "Autograd.hpp"

namespace EngineOps {
    Tensor transpose(const Tensor& A);
    Tensor matmul(const Tensor& A, const Tensor& B);
    Tensor add(const Tensor& A, const Tensor& B);
    Tensor sub(const Tensor& A, const Tensor& B);
    Tensor relu(const Tensor& A);
    Tensor mse_loss(const Tensor& pred, const Tensor& target);
    Tensor gelu(const Tensor& A);
    ValuePtr gelu(const ValuePtr& A);

    ValuePtr transpose(const ValuePtr& A);
    ValuePtr matmul(const ValuePtr& A, const ValuePtr& B);
    ValuePtr add(const ValuePtr& A, const ValuePtr& B);
    ValuePtr sub(const ValuePtr& A, const ValuePtr& B);
    ValuePtr relu(const ValuePtr& A);
    ValuePtr mse_loss(const ValuePtr& pred, const ValuePtr& target);
    Tensor softmax(const Tensor& A, int dim = -1);
    ValuePtr softmax(const ValuePtr& A, int dim = -1);

    Tensor scale(const Tensor& A, float factor);
    ValuePtr scale(const ValuePtr& A, float factor);

    Tensor im2col(const Tensor& input, size_t kernel_h, size_t kernel_w, size_t stride = 1, size_t padding = 0);
    Tensor conv2d(const Tensor& input, const Tensor& kernel, size_t stride = 1, size_t padding = 0);
    ValuePtr conv2d(const ValuePtr& input, const ValuePtr& kernel, size_t stride = 1, size_t padding = 0);

}