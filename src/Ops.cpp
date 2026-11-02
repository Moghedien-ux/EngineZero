    #include "Ops.hpp"
    #include <stdexcept>
    #include <algorithm>

namespace EngineOps {
    Tensor transpose(const Tensor& A) {
        if (A.ndim() != 2) {
            throw std::invalid_argument("Mat not 2D.");
        }

        size_t M = A.shape()[0];
        size_t N = A.shape()[1];

        Tensor A_T({N, M});
        const float* src = A.data();
        float* dst = A_T.data();

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                dst[j * M + i] = src[i * N + j];
            }
        }

        return A_T;
    }

    Tensor matmul(const Tensor& A, const Tensor& B) {
        if (A.ndim() != 2 || B.ndim() != 2) {
            throw std::invalid_argument("Not 2D.");
        }

        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t K_B = B.shape()[0];
        size_t N = B.shape()[1];

        if (K != K_B) {
            throw std::invalid_argument("Not fitting!");
        }

        Tensor C({M, N}, 0.0f);

        const float* a_ptr = A.data();
        const float* b_ptr = B.data();
        float* c_ptr = C.data();

        constexpr size_t BLOCK_SIZE = 64;

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                size_t i_max = std::min(i0 + BLOCK_SIZE, M);
                size_t j_max = std::min(j0 + BLOCK_SIZE, N);

                for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                    size_t k_max = std::min(k0 + BLOCK_SIZE, K);

                    for (size_t i = i0; i < i_max; ++i) {
                        for (size_t k = k0; k < k_max; ++k) {
                            float a_val = a_ptr[i * K + k];

                            #pragma omp simd
                            for (size_t j = j0; j < j_max; ++j) {
                                c_ptr[i * N + j] += a_val * b_ptr[k * N + j];
                            }
                        }
                    }
                }
            }
        }

        return C;
    }

    Tensor add(const Tensor& A, const Tensor& B) {
        if (A.shape() == B.shape()) {
        Tensor C(A.shape());
        size_t total_elements = A.size();
        const float* a_ptr = A.data();
        const float* b_ptr = B.data();
        float* c_ptr = C.data();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total_elements; ++i) {
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }
        return C;
    }

        if (A.ndim() == 2 && B.ndim() == 2 && B.shape()[0] == 1 && A.shape()[1] == B.shape()[1]) {
        size_t M = A.shape()[0];
        size_t N = A.shape()[1];

        Tensor C(A.shape());
        const float* a_ptr = A.data();
        const float* b_ptr = B.data();
        float* c_ptr = C.data();

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                c_ptr[i * N + j] = a_ptr[i * N + j] + b_ptr[j];
            }
        }
        return C;
    }

    throw std::invalid_argument("TENSOR ADD: Incompatibilitate de dimensiuni la adunare!");
}

ValuePtr add(const ValuePtr& A, const ValuePtr& B) {
    Tensor c_data = add(A->data, B->data);
    auto out = ValueNode::make(c_data, {A, B}, "add");

    out->backward_fn = [A, B, out]() {
                if (A->data.shape() != B->data.shape() && B->data.shape()[0] == 1) {
            A->grad = add(A->grad, out->grad);

            size_t M = out->grad.shape()[0];
            size_t N = out->grad.shape()[1];

            Tensor b_grad_reduced({1, N}, 0.0f);
            const float* out_g = out->grad.data();
            float* b_g = b_grad_reduced.data();

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    b_g[j] += out_g[i * N + j];
                }
            }
            B->grad = add(B->grad, b_grad_reduced);
        } else {
            A->grad = add(A->grad, out->grad);
            B->grad = add(B->grad, out->grad);
        }
    };

    return out;
}

    ValuePtr matmul(const ValuePtr& A, const ValuePtr& B) {
        Tensor c_data = matmul(A->data, B->data);
        auto out = ValueNode::make(c_data, {A, B}, "matmul");

        out->backward_fn = [A, B, out]() {
            Tensor grad_A = matmul(out->grad, transpose(B->data));
            A->grad = add(A->grad, grad_A);

            Tensor grad_B = matmul(transpose(A->data), out->grad);
            B->grad = add(B->grad, grad_B);
        };

        return out;
    }

    ValuePtr transpose(const ValuePtr& A) {
        Tensor t_data = transpose(A->data);
        auto out = ValueNode::make(t_data, {A}, "transpose");

        out->backward_fn = [A, out]() {
            A->grad = add(A->grad, transpose(out->grad));
        };

        return out;
    }

        Tensor relu(const Tensor& A) {
        Tensor C(A.shape());
        size_t total = A.size();
        const float* src = A.data();
        float* dst = C.data();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total; ++i) {
            dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
        }

        return C;
    }

        ValuePtr relu(const ValuePtr& A) {
        Tensor c_data = relu(A->data);
        auto out = ValueNode::make(c_data, {A}, "relu");

        out->backward_fn = [A, out]() {
            size_t total = A->data.size();
            const float* a_data = A->data.data();
            const float* out_grad = out->grad.data();
            float* a_grad = A->grad.data();

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total; ++i) {
                if (a_data[i] > 0.0f) {
                    a_grad[i] += out_grad[i];
                }
            }
        };

        return out;
    }

    Tensor sub(const Tensor& A, const Tensor& B) {
        if (A.shape() != B.shape()) {
            throw std::invalid_argument("TENSOR SUB: Shape-urile trebuie sa fie identice!");
        }

        Tensor C(A.shape());
        size_t total = A.size();
        const float* a_ptr = A.data();
        const float* b_ptr = B.data();
        float* c_ptr = C.data();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total; ++i) {
            c_ptr[i] = a_ptr[i] - b_ptr[i];
        }

        return C;
    }

    ValuePtr sub(const ValuePtr& A, const ValuePtr& B) {
        Tensor c_data = sub(A->data, B->data);
        auto out = ValueNode::make(c_data, {A, B}, "sub");

        out->backward_fn = [A, B, out]() {
            A->grad = add(A->grad, out->grad);

                        Tensor neg_grad(out->grad.shape());
            size_t total = out->grad.size();
            for (size_t i = 0; i < total; ++i) {
                neg_grad.at_flat(i) = -out->grad.at_flat(i);
            }
            B->grad = add(B->grad, neg_grad);
        };

        return out;
    }

    Tensor mse_loss(const Tensor& pred, const Tensor& target) {
        if (pred.shape() != target.shape()) {
            throw std::invalid_argument("MSE LOSS: Pred si Target trebuie sa aiba aceeasi forma!");
        }

        float sum_sq_err = 0.0f;
        size_t total = pred.size();
        const float* p_ptr = pred.data();
        const float* t_ptr = target.data();

        #pragma omp parallel for reduction(+:sum_sq_err) schedule(static)
        for (size_t i = 0; i < total; ++i) {
            float diff = p_ptr[i] - t_ptr[i];
            sum_sq_err += diff * diff;
        }

        return Tensor({1, 1}, sum_sq_err / static_cast<float>(total));
    }

    ValuePtr mse_loss(const ValuePtr& pred, const ValuePtr& target) {
        Tensor loss_data = mse_loss(pred->data, target->data);
        auto out = ValueNode::make(loss_data, {pred, target}, "mse_loss");

        out->backward_fn = [pred, target, out]() {
            size_t total = pred->data.size();
            float scale = 2.0f / static_cast<float>(total);

            Tensor grad_pred(pred->data.shape());
            const float* p_ptr = pred->data.data();
            const float* t_ptr = target->data.data();
            float* g_ptr = grad_pred.data();

            float out_g = out->grad.at_flat(0);

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total; ++i) {
                g_ptr[i] = scale * (p_ptr[i] - t_ptr[i]) * out_g;
            }

            pred->grad = add(pred->grad, grad_pred);
        };

        return out;
    }

    Tensor gelu(const Tensor& A) {
        Tensor C(A.shape());
        size_t total = A.size();
        const float* src = A.data();
        float* dst = C.data();

        constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total; ++i) {
            float x = src[i];
            float cube = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + 0.044715f * cube);
            dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }

        return C;
    }

    ValuePtr gelu(const ValuePtr& A) {
        Tensor c_data = gelu(A->data);
        auto out = ValueNode::make(c_data, {A}, "gelu");

        out->backward_fn = [A, out]() {
            size_t total = A->data.size();
            const float* x_ptr = A->data.data();
            const float* out_grad = out->grad.data();
            float* a_grad = A->grad.data();

            constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total; ++i) {
                float x = x_ptr[i];
                float x2 = x * x;
                float x3 = x2 * x;
                float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
                float tanh_inner = std::tanh(inner);

                float sech2 = 1.0f - tanh_inner * tanh_inner;
                float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * 0.044715f * x2);
                float cdf = 0.5f * (1.0f + tanh_inner);
                float pdf = 0.5f * x * sech2 * d_inner;

                a_grad[i] += (cdf + pdf) * out_grad[i];
            }
        };

        return out;
    }

    #include <cmath>
#include <limits>

    Tensor softmax(const Tensor& A, int dim) {
        if (A.ndim() != 2) {
            throw std::invalid_argument("Softmax este optimizat pentru tensori 2D [Batch, Features].");
        }

        size_t rows = A.shape()[0];
        size_t cols = A.shape()[1];

        Tensor C(A.shape());
        const float* src = A.data();
        float* dst = C.data();

        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r) {
            size_t row_offset = r * cols;

                        float max_val = -std::numeric_limits<float>::infinity();
            for (size_t c = 0; c < cols; ++c) {
                if (src[row_offset + c] > max_val) {
                    max_val = src[row_offset + c];
                }
            }

                        float sum_exp = 0.0f;
            for (size_t c = 0; c < cols; ++c) {
                float e = std::exp(src[row_offset + c] - max_val);
                dst[row_offset + c] = e;
                sum_exp += e;
            }

                        float inv_sum = 1.0f / sum_exp;
            for (size_t c = 0; c < cols; ++c) {
                dst[row_offset + c] *= inv_sum;
            }
        }

        return C;
    }

    ValuePtr softmax(const ValuePtr& A, int dim) {
        Tensor c_data = softmax(A->data, dim);
        auto out = ValueNode::make(c_data, {A}, "softmax");

        out->backward_fn = [A, out]() {
            size_t rows = A->data.shape()[0];
            size_t cols = A->data.shape()[1];

            const float* s_ptr = out->data.data();                 const float* dL_dS = out->grad.data();                float* dL_dX = A->grad.data();
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < rows; ++r) {
                size_t row_offset = r * cols;

                                float sum_grad_s = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    size_t idx = row_offset + c;
                    sum_grad_s += dL_dS[idx] * s_ptr[idx];
                }

                                for (size_t c = 0; c < cols; ++c) {
                    size_t idx = row_offset + c;
                    dL_dX[idx] += s_ptr[idx] * (dL_dS[idx] - sum_grad_s);
                }
            }
        };

        return out;
    }

    Tensor scale(const Tensor& A, float factor) {
        Tensor C(A.shape());
        size_t total = A.size();
        const float* src = A.data();
        float* dst = C.data();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total; ++i) {
            dst[i] = src[i] * factor;
        }

        return C;
    }

    ValuePtr scale(const ValuePtr& A, float factor) {
        Tensor c_data = scale(A->data, factor);
        auto out = ValueNode::make(c_data, {A}, "scale");

        out->backward_fn = [A, factor, out]() {
            size_t total = A->data.size();
            const float* out_g = out->grad.data();
            float* a_g = A->grad.data();

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total; ++i) {
                a_g[i] += out_g[i] * factor;
            }
        };

        return out;
    }

    Tensor im2col(const Tensor& input, size_t kernel_h, size_t kernel_w, size_t stride, size_t padding) {
        size_t C = input.shape()[0];
    size_t H = input.shape()[1];
    size_t W = input.shape()[2];

    size_t out_h = (H + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (W + 2 * padding - kernel_w) / stride + 1;

    Tensor col({C * kernel_h * kernel_w, out_h * out_w}, 0.0f);
    float* col_data = col.data();
    const float* in_data = input.data();

    for (size_t c = 0; c < C; ++c) {
        for (size_t kh = 0; kh < kernel_h; ++kh) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
                size_t row_idx = c * (kernel_h * kernel_w) + kh * kernel_w + kw;

                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                        int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                        float val = 0.0f;
                        if (ih >= 0 && ih < static_cast<int>(H) && iw >= 0 && iw < static_cast<int>(W)) {
                            val = in_data[c * (H * W) + ih * W + iw];
                        }

                        size_t col_idx = oh * out_w + ow;
                        col_data[row_idx * (out_h * out_w) + col_idx] = val;
                    }
                }
            }
        }
    }

    return col;
    }

    Tensor conv2d(const Tensor& input, const Tensor& kernel, size_t stride, size_t padding) {
                Tensor col = im2col(input, 3, 3, stride, padding);         return matmul(kernel, col);
    }

    ValuePtr conv2d(const ValuePtr& input, const ValuePtr& kernel, size_t stride, size_t padding) {
        Tensor out_data = conv2d(input->data, kernel->data, stride, padding);
        auto out = ValueNode::make(out_data, {input, kernel}, "conv2d");

        out->backward_fn = [input, kernel, out]() {
            Tensor col = im2col(input->data, 3, 3, 1, 0);
            Tensor grad_k = matmul(out->grad, transpose(col));
            kernel->grad = add(kernel->grad, grad_k);
        };

        return out;
}

}