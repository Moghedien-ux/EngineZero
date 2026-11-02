#pragma once

#include <vector>
#include <initializer_list>
#include <cstddef>
#include <memory>
#include <iostream>
#include <concepts>

constexpr size_t ALIGNMENT_BYTES = 64;

class Tensor {

    float *m_data{nullptr};
    size_t m_size{0};
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;

    void allocate_memory();
    void compute_strides();

public:
    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, float fill_value);
    Tensor(std::initializer_list<size_t> shape);

    ~Tensor();
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

        [[nodiscard]] inline float* data() noexcept {return m_data;}
    [[nodiscard]] inline const float* data() const noexcept {return m_data;}
    [[nodiscard]] inline size_t size() const noexcept {return m_size;}
    [[nodiscard]] inline const std::vector<size_t>& shape() const noexcept {return m_shape;}
    [[nodiscard]] inline const std::vector<size_t>& strides() const noexcept {return m_strides;}
    [[nodiscard]] inline size_t ndim() const noexcept {return m_shape.size();}

        inline float& at_flat(size_t index) noexcept {return m_data[index];}
    [[nodiscard]] inline const float& at_flat(size_t index) const noexcept {return m_data[index];}

    template<std::integral... Args>
    [[nodiscard]] inline float& operator[](Args... indices) {
        size_t flat_idx = 0;
        size_t i = 0;
        ((flat_idx += static_cast<size_t>(indices) * m_strides[i++]), ...);
        return m_data[flat_idx];
    }

    template<std::integral... Args>
    [[nodiscard]] inline const float& operator[](Args... indices) const {
        size_t flat_idx = 0;
        size_t i = 0;
        ((flat_idx += static_cast<size_t>(indices) * m_strides[i++]), ...);
        return m_data[flat_idx];
    }

    void print() const;
};
